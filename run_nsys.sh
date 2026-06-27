#!/usr/bin/env bash
# Nsight Systems profiling — system-level timeline (CUDA kernels + NVTX ranges)
#
# Output: profiles/nsys/{baseline,nanovllm}.nsys-rep
# Viewer:   nsys-ui profiles/nsys/nanovllm.nsys-rep
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

COMMON_FLAGS=(
    --trace=cuda,nvtx,osrt
    --cuda-memory-usage=true     # track GPU memory alloc/free over time
    --backtrace=none             # skip CPU backtraces (much faster capture)
    --force-overwrite=true
    --capture-range=cudaProfilerApi   # gate: cudaProfilerStart/Stop in Python
    --capture-range-end=stop          # stop collection when Stop is called
)

echo "================================================================"
echo " [1/2] Profiling baseline (HF Transformers, no paged attention)"
echo "================================================================"
$NSYS profile \
    "${COMMON_FLAGS[@]}" \
    --output="$OUTDIR/baseline" \
    python "$(dirname "$0")/baseline.py"

echo ""
echo "================================================================"
echo " [2/2] Profiling nano-vLLM (paged attention + CUDA graphs)"
echo "================================================================"
$NSYS profile \
    "${COMMON_FLAGS[@]}" \
    --output="$OUTDIR/nanovllm" \
    python "$(dirname "$0")/nanovllm_bench.py"

echo ""
echo "Done. Open with:"
echo "  nsys-ui $OUTDIR/baseline.nsys-rep"
echo "  nsys-ui $OUTDIR/nanovllm.nsys-rep"
