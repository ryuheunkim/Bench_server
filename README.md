# nano-vLLM Paged Attention 프로파일링 벤치마크

nano-vLLM의 paged attention + CUDA Graph가 GPU 메모리를 어떻게 관리하는지,
HuggingFace Transformers 기본 KV 캐시와 비교하는 프로파일링 환경입니다.
NVTX 마커를 monkey-patch 방식으로 삽입하고 Nsight Systems로 타임라인을 추출합니다.

---

## 설치 방법

```bash
# 1. 환경 생성 및 활성화
conda create -n nanovllm python=3.10 -y
conda activate nanovllm

# 2. PyTorch (CUDA 12.1 빌드)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. flash-attn (빌드 포함, 5~10분 소요)
pip install flash-attn==2.8.3.post1 --no-build-isolation

# 4. 나머지 패키지 (nano-vllm 포함)
pip install -r requirements.txt
```

> **순서 주의:** `flash-attn`을 먼저 설치해야 합니다. `nano-vllm`이 설치 시점에 flash-attn 존재 여부를 확인하기 때문입니다.

---

## 실행 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 4050 (VRAM 6 GB) |
| CPU | AMD Ryzen 5 8645HS |
| RAM | 16 GB |
| OS | Ubuntu (Linux 6.8) |
| Python | 3.10 (conda 환경: `nanovllm`) |
| PyTorch | CUDA 12.4 |
| 모델 | Qwen3-0.6B (fp16, `~/huggingface/Qwen3-0.6B`) |
| nano-vLLM 소스 | `~/nano-vllm` |

---

## 파일 구성

```
Bench_server/
  baseline.py          HF Transformers — 표준 KV 캐시 (non-paged)
  nanovllm_bench.py    nano-vLLM — paged attention + CUDA Graph, NVTX 포함
  run_nsys.sh          Nsight Systems 실행 스크립트
  run_ncu.sh           Nsight Compute 실행 스크립트
  profiles/
    nsys/
      baseline.nsys-rep
      nanovllm.nsys-rep
```

---

## 주요 벤치마크 파라미터

두 스크립트에서 동일한 값을 사용합니다.

| 파라미터 | 값 | 설명 |
|---|---|---|
| `NUM_SEQS` | 16 | 총 시퀀스 수 |
| `MAX_INPUT` | 128 tok | 입력 길이 상한 (randint 32~128) |
| `MAX_OUTPUT` | 64 tok | 출력 길이 상한 (randint 32~64) |
| `VOCAB_SIZE` | 151936 | Qwen3 어휘 크기 (랜덤 토큰 생성용) |

**파라미터 선정 이유 (RTX 4050 기준):**
- HF 배치 처리 시 attention matrix 크기: `[B, H, L, L]` fp32
  - `[16, 16, 192, 192] × 4B ≈ 144 MB` — VRAM 여유 있음
- `NUM_SEQS=32` 이상 또는 `MAX_INPUT=256` 이상에서 OOM 발생 확인

---

## 실행 방법

```bash
conda activate nanovllm
cd ~/KNOU/Bench_server

# Nsight Systems (CUDA 커널 타임라인)
./run_nsys.sh

# Nsight Compute (커널별 상세 메트릭)
./run_ncu.sh

# 결과 열기
nsys-ui profiles/nsys/baseline.nsys-rep
nsys-ui profiles/nsys/nanovllm.nsys-rep
```

---

## 두 구현체의 메모리 모델 비교

### baseline (HF Transformers)

- 16개 시퀀스를 좌측 패딩 후 단일 텐서 `[16, L_max]`로 배치
- KV 캐시: 배치 전체를 하나의 연속 텐서로 할당 → 시퀀스 간 공유 불가
- 출력 길이가 다른 시퀀스도 `max(max_new_tokens)` 스텝을 모두 실행 → 낭비
- attention 구현: eager (standard scaled-dot-product, flash 없음)

### nanovllm_bench (nano-vLLM)

- KV 캐시: 물리 블록 풀 사전 할당 `[2, layers, num_blocks, block_size, kv_heads, head_dim]`
- 블록을 시퀀스와 독립적으로 관리 → prefix cache 재사용 가능
- prefill: `flash_attn_varlen_func` (가변 길이, cu_seqlens 경계)
- decode: `flash_attn_with_kvcache` (block_table 기반 비연속 KV 접근)
- CUDA Graph: decode batch size별 그래프 캡처 → Python overhead 제거

---

## NVTX 마커 (nanovllm_bench.py)

monkey-patch 원칙: `LLM()` 인스턴스 생성 **전에** 모듈 attribute 교체.

| 마커 | 패치 대상 | 타임라인에서 보여주는 것 |
|---|---|---|
| `store_kvcache[triton_scatter]` | `_attn.store_kvcache` | slot_mapping scatter → L2 miss 패턴 |
| `flash_attn[prefill_varlen]` | `Attention.forward` | varlen prefill 커널 |
| `flash_attn[decode_paged]` | `Attention.forward` | block_table 기반 비연속 KV 읽기 |
| `block_alloc[cached=N,new=M]` | `BlockManager.allocate` | prefix cache 히트율 |
| `block_dealloc` | `BlockManager.deallocate` | 블록 반납 타이밍 |
| `cudagraph_replay[bs=N]` | `ModelRunner.run_model` | CUDA Graph decode overhead 제거 효과 |
| `benchmark_generate` | 최상위 with 블록 | nsys `--capture-range` 게이트 |

---

## nsys 캡처 설정 (`run_nsys.sh`)

```
--trace=cuda,nvtx,osrt
--cuda-memory-usage=true
--capture-range=nvtx
--nvtx-capture="benchmark_generate"
```

- 두 스크립트 모두 `benchmark_generate` NVTX 범위 안에서만 캡처
- 모델 로딩·워밍업 구간은 트레이싱 제외 → `.nsys-rep` 파일 크기 수백 MB 이내 유지
- `--backtrace=none`: CPU 백트레이스 생략으로 캡처 오버헤드 최소화

### ncu 주의사항 (`run_ncu.sh`)

```
--replay-mode application
```

CUDA Graph와 kernel-replay 모드 충돌 방지. 이 옵션 없이 실행하면 프로파일이 비거나 틀린 결과가 나옴.

---

## torch._dynamo 캐시 설정

`nanovllm_bench.py`에서 `LLM()` 생성 전:

```python
torch._dynamo.config.cache_size_limit = 64
```

CUDA Graph 캡처 시 batch size별(1, 2, 4, 8, ..., 512) `rms_forward` 재컴파일이 발생하는데,
기본 한도(8)를 초과하면 매 호출마다 재컴파일 → GPU 과부하. 64로 늘려서 캡처 완료까지 캐시 유지.
