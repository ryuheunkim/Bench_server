"""
nano-vLLM benchmark with NVTX instrumentation for Nsight Systems / Nsight Compute.

Monkey-patching strategy:
  - All patches are applied to module-level names BEFORE LLM() is constructed.
  - Works for single-GPU (tp=1) because ModelRunner runs inline in rank-0 process.

NVTX ranges added:
  store_kvcache[triton_scatter]  — Triton kernel that scatters K/V into paged blocks
  flash_attn[prefill_varlen]     — flash_attn_varlen_func (variable-length prefill)
  flash_attn[decode_paged]       — flash_attn_with_kvcache (decode via block_table)
  block_alloc[cached=N,new=M]    — BlockManager allocating N reused + M fresh blocks
  block_dealloc                  — BlockManager returning blocks to free-list
  model_run[prefill|decode,...]  — one ModelRunner.run() call
  cudagraph_replay[bs=N]         — CUDAGraph.replay() for decode (no Python overhead)
  eager_forward[...]             — model forward without CUDAGraph
  warmup / benchmark_generate    — top-level timing gates for nsys --capture-range
"""

import os
import sys
import time
import torch
from random import randint, seed

sys.path.insert(0, os.path.expanduser("~/nano-vllm"))

NVTX = torch.cuda.nvtx
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")
NUM_SEQS   = 32
MAX_INPUT  = 128
MAX_OUTPUT = 64
VOCAB_SIZE = 151936  # Qwen3


# ── NVTX context manager ──────────────────────────────────────────────────────

class nvtx_range:
    __slots__ = ("label",)
    def __init__(self, label): self.label = label
    def __enter__(self): NVTX.range_push(self.label)
    def __exit__(self, *_): NVTX.range_pop()


# ── Monkey-patches (applied before any LLM object is created) ─────────────────

import nanovllm.layers.attention  as _attn
import nanovllm.engine.block_manager as _bm
import nanovllm.engine.model_runner  as _mr

# 1. store_kvcache
#    slot_mapping[i] = physical_block_id * block_size + offset_within_block
#    The Triton kernel uses this indirection to scatter K/V into the paged pool.
#    Non-contiguous writes → watch for L2 cache pressure in Nsight Compute.
_orig_store_kvcache = _attn.store_kvcache
def _store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    with nvtx_range("store_kvcache[triton_scatter]"):
        return _orig_store_kvcache(key, value, k_cache, v_cache, slot_mapping)
_attn.store_kvcache = _store_kvcache

# 2. Attention.forward
#    Prefill  → flash_attn_varlen_func  (block_tables=None or prefix-cache)
#    Decode   → flash_attn_with_kvcache (block_tables = per-seq page map)
_orig_attn_fwd = _attn.Attention.forward
def _attn_fwd(self, q, k, v):
    from nanovllm.utils.context import get_context
    phase = "prefill_varlen" if get_context().is_prefill else "decode_paged"
    with nvtx_range(f"flash_attn[{phase}]"):
        return _orig_attn_fwd(self, q, k, v)
_attn.Attention.forward = _attn_fwd

# 3. BlockManager.allocate
#    num_cached_blocks > 0  → prefix cache hit: blocks are reused (ref_count++)
#    new blocks             → popped from free_block_ids deque
_orig_alloc = _bm.BlockManager.allocate
def _alloc(self, seq, num_cached_blocks):
    new_blocks = seq.num_blocks - num_cached_blocks
    with nvtx_range(f"block_alloc[cached={num_cached_blocks},new={new_blocks}]"):
        _orig_alloc(self, seq, num_cached_blocks)
_bm.BlockManager.allocate = _alloc

# 4. BlockManager.deallocate
#    ref_count-- per block; block returned to deque when ref_count == 0
_orig_dealloc = _bm.BlockManager.deallocate
def _dealloc(self, seq):
    with nvtx_range("block_dealloc"):
        _orig_dealloc(self, seq)
_bm.BlockManager.deallocate = _dealloc

# 5. ModelRunner.run (one scheduler step)
_orig_run = _mr.ModelRunner.run
def _run(self, seqs, is_prefill):
    phase = "prefill" if is_prefill else "decode"
    toks  = sum(s.num_scheduled_tokens for s in seqs) if is_prefill else len(seqs)
    with nvtx_range(f"model_run[{phase},seqs={len(seqs)},toks={toks}]"):
        return _orig_run(self, seqs, is_prefill)
_mr.ModelRunner.run = _run

# 6. ModelRunner.run_model (distinguish CUDA graph replay vs eager)
_orig_run_model = _mr.ModelRunner.run_model
def _run_model(self, input_ids, positions, is_prefill):
    bs = input_ids.size(0)
    if not is_prefill and not self.enforce_eager and bs <= 512:
        label = f"cudagraph_replay[bs={bs}]"
    else:
        label = f"eager_forward[prefill={is_prefill},bs={bs}]"
    with nvtx_range(label):
        return _orig_run_model(self, input_ids, positions, is_prefill)
_mr.ModelRunner.run_model = _run_model


# ── Benchmark ─────────────────────────────────────────────────────────────────

from nanovllm import LLM, SamplingParams


def main():
    seed(0)

    # rms_forward is recompiled once per CUDA Graph batch size (up to ~20 sizes).
    # Default cache_size_limit=8 causes recompilation storms during graph capture.
    torch._dynamo.config.cache_size_limit = 64

    with nvtx_range("llm_init"):
        llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, VOCAB_SIZE - 1) for _ in range(randint(32, MAX_INPUT))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(32, MAX_OUTPUT))
        for _ in range(NUM_SEQS)
    ]

    # ── Warmup ────────────────────────────────────────────────────────────────
    # cudaProfilerStart is called after warmup, so nsys skips this section.
    with nvtx_range("warmup"):
        llm.generate(["Benchmark: "], SamplingParams())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ── Timed benchmark ───────────────────────────────────────────────────────
    # cudaProfilerStart gates nsys --capture-range=cudaProfilerApi.
    # NVTX range is kept so the marker appears in the timeline too.
    torch.cuda.cudart().cudaProfilerStart()
    with nvtx_range("benchmark_generate"):
        t_start = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
    torch.cuda.cudart().cudaProfilerStop()

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    peak_mem_gb  = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*60}")
    print(f"nano-vLLM (paged attention, prefix cache, CUDA graphs)")
    print(f"  Sequences : {NUM_SEQS}")
    print(f"  Max input : {MAX_INPUT} tok  |  Max output: {MAX_OUTPUT} tok")
    print(f"  Total out : {total_tokens} tok")
    print(f"  Time      : {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens / elapsed:.1f} tok/s")
    print(f"  Peak VRAM : {peak_mem_gb:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
