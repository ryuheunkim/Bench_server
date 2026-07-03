#!/usr/bin/env bash
# Nsight Systems profiling — system-level timeline (CUDA kernels + NVTX ranges)
#
# Output: profiles/nsys/{baseline,nanovllm}_YYYYMMDD_HHMMSS.nsys-rep
# Viewer:   nsys-ui profiles/nsys/nanovllm_<date>.nsys-rep
#
# Key NVTX ranges to look for in the timeline:
#   benchmark_generate        — entire timed region
#   model_run[prefill,...]    — one scheduler step (prefill)
#   model_run[decode,...]     — one scheduler step (decode)
#   flash_attn[prefill_varlen]— flash_attn_varlen_func kernel call
#   flash_attn[decode_paged]  — flash_attn_with_kvcache kernel call
#   store_kvcache[...]        — Triton scatter into paged KV pool
#   block_alloc[...]          — BlockManager.allocate (CPU only, no GPU)
#   cudagraph_replay[bs=N]    — CUDAGraph.replay() for decode batches

set -euo pipefail

NSYS=/usr/local/bin/nsys
OUTDIR="$(dirname "$0")/profiles/nsys"
mkdir -p "$OUTDIR"

DATE=$(date +%Y%m%d_%H%M%S)

COMMON_FLAGS=(
    --trace=cuda,nvtx,osrt
    --gpu-metrics-devices=all
    --cuda-memory-usage=true
    --backtrace=none
    --force-overwrite=true
)

echo "================================================================"
echo " [1/2] Profiling baseline (HF Transformers, no paged attention)"
echo "================================================================"
# capture-range 없음: model load부터 전체 캡처 → osrt cold-start overhead 방지
$NSYS profile \
    "${COMMON_FLAGS[@]}" \
    --output="$OUTDIR/baseline_${DATE}" \
    python "$(dirname "$0")/baseline.py"

echo ""
echo "================================================================"
echo " [2/2] Profiling nano-vLLM (paged attention + CUDA graphs)"
echo "================================================================"
# cudaProfilerApi로 게이팅: model load / CUDA graph capture 제외하고 inference만 캡처
$NSYS profile \
    "${COMMON_FLAGS[@]}" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="$OUTDIR/nanovllm_${DATE}" \
    python "$(dirname "$0")/nanovllm_bench.py"

echo ""
echo "Done. Open with:"
echo "  nsys-ui $OUTDIR/baseline_${DATE}.nsys-rep"
echo "  nsys-ui $OUTDIR/nanovllm_${DATE}.nsys-rep"
