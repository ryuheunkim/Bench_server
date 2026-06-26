#!/usr/bin/env bash
# Nsight Compute profiling — per-kernel hardware metrics
#
# Output: profiles/ncu/{nanovllm,baseline}_kernels.ncu-rep
# Viewer:   ncu-ui profiles/ncu/nanovllm_kernels.ncu-rep
#
# Why --replay-mode application:
#   - nano-vLLM uses CUDA graphs for decode; kernel-replay mode interferes
#     with graph capture/replay, producing empty or incorrect profiles.
#   - Application-replay re-runs the full program for each metric pass,
#     so CUDA graphs work correctly at the cost of longer profiling time.
#
# Targeted kernels (nano-vLLM):
#   store_kvcache_kernel_*   — Triton scatter K/V into paged blocks
#   flash_fwd_varlen_*       — prefill attention (variable-length sequences)
#   flash_fwd_*kvcache*      — decode attention reading via block_table
#
# Key metrics for GPU memory analysis:
#   MemoryWorkloadAnalysis   — L1/L2/DRAM throughput, hit rates, access patterns
#   SpeedOfLight             — SM and memory utilisation vs theoretical peak
#   ComputeWorkloadAnalysis  — warp efficiency, occupancy

set -euo pipefail

NCU=/usr/local/cuda-12.9/bin/ncu
OUTDIR="$(dirname "$0")/profiles/ncu"
mkdir -p "$OUTDIR"

COMMON_FLAGS=(
    --target-processes all
    --replay-mode application        # required for CUDA-graph workloads
    --kernel-name-base demangled
    --section MemoryWorkloadAnalysis
    --section SpeedOfLight
    --section ComputeWorkloadAnalysis
    --force-overwrite
)

echo "================================================================"
echo " [1/2] Kernel profiling: nano-vLLM paged-attention kernels"
echo "================================================================"
# Profile store_kvcache (Triton) + flash_fwd (prefill + decode variants)
$NCU \
    "${COMMON_FLAGS[@]}" \
    --kernel-name "regex:(flash_fwd|store_kvcache)" \
    --output "$OUTDIR/nanovllm_kernels" \
    python "$(dirname "$0")/nanovllm_bench.py"

echo ""
echo "================================================================"
echo " [2/2] Kernel profiling: baseline standard-attention kernels"
echo "================================================================"
# HF eager attention uses scaled_dot_product_attention → cudnn / aten kernels
$NCU \
    "${COMMON_FLAGS[@]}" \
    --kernel-name "regex:(attention|softmax_warp|scaled_dot)" \
    --output "$OUTDIR/baseline_kernels" \
    python "$(dirname "$0")/baseline.py"

echo ""
echo "Done. Open with:"
echo "  ncu-ui $OUTDIR/nanovllm_kernels.ncu-rep"
echo "  ncu-ui $OUTDIR/baseline_kernels.ncu-rep"

# ── Quick single-pass comparison (optional) ───────────────────────────────────
# Uncomment to get a fast summary without full section analysis:
#
# $NCU \
#     --target-processes all \
#     --replay-mode application \
#     --kernel-name "regex:(flash_fwd|store_kvcache)" \
#     --metrics \
#         "sm__throughput.avg.pct_of_peak_sustained_elapsed,\
#          lts__average_t_sector_hit_rate.pct,\
#          dram__bytes_read.sum,\
#          dram__bytes_write.sum,\
#          gpu__time_duration.sum" \
#     --csv \
#     python "$(dirname "$0")/nanovllm_bench.py" \
#     > "$OUTDIR/nanovllm_quick.csv"
