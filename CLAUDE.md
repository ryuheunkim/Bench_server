# nano-vLLM — Claude Code Context

## 진행 중인 작업

**목표:** nano-vLLM의 paged attention 메커니즘이 GPU 메모리를 어떻게 관리하는지
코드와 프로파일링 결과를 매칭해서 보고서 작성.

**프로파일링 스크립트 위치:** `~/bench-nanovllm/` (이 repo 밖)

| 파일 | 역할 |
|---|---|
| `baseline.py` | HF Transformers — 표준 KV 캐시, non-paged |
| `nanovllm_bench.py` | nano-vLLM + NVTX monkey-patch 벤치마크 |
| `run_nsys.sh` | Nsight Systems 실행 (`nsys profile`) |
| `run_ncu.sh` | Nsight Compute 실행 (`ncu`) |

실행: `cd ~/bench-nanovllm && ./run_nsys.sh` 또는 `./run_ncu.sh`

nano-vLLM 소스를 수정하지 않고 monkey-patching으로 NVTX 마커를 삽입해서,
Nsight에서 코드 위치(`block_alloc`, `store_kvcache` 등)와 GPU 커널이 1:1 매핑되는 타임라인을 얻는 방식.

---

## 코드 구조 메모

```
nanovllm/
  layers/
    attention.py      — Triton store_kvcache_kernel + Attention.forward (flash_attn 호출)
  engine/
    block_manager.py  — BlockManager: 물리 블록 풀 관리, prefix cache (xxhash 기반)
    model_runner.py   — ModelRunner: KV 캐시 할당, CUDA Graph capture/replay, prefill/decode prepare
    llm_engine.py     — LLMEngine: step() 루프, scheduler와 model_runner 연결
    scheduler.py      — Scheduler: waiting/running 큐, prefill/decode 스케줄링, preemption
    sequence.py       — Sequence: block_table, num_cached_tokens, num_scheduled_tokens
```

### KV 캐시 레이아웃 (`model_runner.py:103`)

```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
# shape: [K/V, layer, block_id, token_in_block, head, dim]
```
- 전체 KV 풀을 하나의 텐서로 사전 할당.
- 물리 블록(block_id)은 시퀀스와 무관하게 공유 가능.
- 각 Attention 레이어: `module.k_cache = kv_cache[0, layer_id]` 슬라이스 참조.

### BlockManager (`block_manager.py`)

- `free_block_ids: deque` — 사용 가능한 블록 ID FIFO 큐
- `hash_to_block_id: dict` — prefix cache: 토큰 시퀀스 해시 → 블록 ID (xxhash64, 체인 해시)
- `can_allocate(seq)` → prefix 히트 수 반환 (-1이면 OOM)
- `allocate(seq, num_cached_blocks)` → prefix 블록 ref_count++, 신규 블록 deque에서 pop
- `deallocate(seq)` → ref_count-- 후 0이면 deque에 반납 (블록 내용 유지 → lazy eviction)
- `hash_blocks(seq)` → prefill 후 완성된 블록에 해시 등록 (다음 시퀀스가 재사용 가능)

### store_kvcache Triton 커널 (`attention.py:10-40`)

```python
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)   # = block_id * block_size + offset
    if slot == -1: return                    # cached token, skip
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
```
- `slot_mapping`: CPU에서 미리 계산된 물리 주소 배열 (비연속) → scatter 패턴 → L2 cache pressure
- prefill/decode 모두 이 커널로 K/V를 paged pool에 기록

### Attention.forward (`attention.py:59-75`)

- **Prefill**: `flash_attn_varlen_func` — 가변 길이, `cu_seqlens_q/k`로 경계 지정.
  prefix cache 히트 시 `block_tables` 전달 → KV를 paged pool에서 읽음
- **Decode**: `flash_attn_with_kvcache` — `block_table` per-seq 페이지 맵으로 비연속 KV 접근

### CUDA Graph (`model_runner.py:222`)

- batch size [1, 2, 4, 8, 16, 32, ..., 512] 각각 graph 캡처
- decode 시 `graph.replay()` → Python overhead 제거
- `enforce_eager=True` 또는 `bs > 512`이면 미사용

### Scheduler (`scheduler.py:25`)

1. waiting 큐 → prefill 가능한 시퀀스 선택 (`max_num_batched_tokens` 내에서)
2. 없으면 running 큐 → decode 배치 구성
3. 메모리 부족 시 `preempt()` → running → waiting 강등, 블록 반납

---

## 벤치마크 스크립트 설계 결정

### NVTX 삽입 방식

`torch.cuda.nvtx.range_push/pop` (PyTorch 내장, 별도 패키지 불필요).

monkey-patch 원칙: `LLM()` 인스턴스 생성 **전에** 모듈 attribute 교체.
Python 함수는 호출 시점에 모듈 global dict에서 이름을 조회하므로, attribute 교체만으로 소스 수정 없이 wrapping 가능.

### NVTX 마커 위치

| 마커 | 패치 대상 | 보고서에서 보여줄 것 |
|---|---|---|
| `store_kvcache[triton_scatter]` | `_attn.store_kvcache` | slot_mapping 비연속 scatter → L2 miss |
| `flash_attn[prefill_varlen]` | `Attention.forward` | varlen prefill 커널 |
| `flash_attn[decode_paged]` | `Attention.forward` | block_table 기반 비연속 KV 읽기 |
| `block_alloc[cached=N,new=M]` | `BlockManager.allocate` | prefix cache 히트율 |
| `block_dealloc` | `BlockManager.deallocate` | 블록 반납 타이밍 |
| `cudagraph_replay[bs=N]` | `ModelRunner.run_model` | CUDA Graph decode overhead 제거 |
| `benchmark_generate` | 최상위 with 블록 | nsys `--capture-range` 게이트 |

### 비교 파라미터

- baseline: `NUM_SEQS=16, MAX_INPUT=256, MAX_OUTPUT=128` (OOM 방지)
- nanovllm: `NUM_SEQS=256, MAX_INPUT=1024, MAX_OUTPUT=1024` (원본 bench.py와 동일)
- 모델: `~/huggingface/Qwen3-0.6B/`

### ncu 주의사항

`--replay-mode application` 필수.
이유: CUDA Graph가 kernel-replay 모드와 충돌 → 프로파일이 비거나 틀려짐.

마커 추가/제거 시 monkey-patch 순서(import 후, `LLM()` 전)를 반드시 지킬 것.
