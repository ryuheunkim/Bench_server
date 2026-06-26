"""
Baseline: HuggingFace Transformers — standard (non-paged) KV cache.

Memory model:
  - KV cache is a contiguous tensor per sequence, pre-allocated to max_new_tokens.
  - All sequences in a batch are padded to the same length → memory wasted on padding.
  - No block sharing, no prefix caching, no on-demand allocation.

Compare with nanovllm_bench.py to see the effect of paged attention.
"""

import os
import time
import torch
from random import randint, seed
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
NUM_SEQS    = 16     # small: standard KV cache OOMs at the scale nano-vLLM handles
MAX_INPUT   = 256
MAX_OUTPUT  = 128
VOCAB_SIZE  = 151936  # Qwen3

NVTX = torch.cuda.nvtx


class nvtx_range:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        NVTX.range_push(self.label)
    def __exit__(self, *_):
        NVTX.range_pop()


def main():
    seed(0)

    # ── Load model ────────────────────────────────────────────────────────────
    with nvtx_range("model_load"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            attn_implementation="eager",   # standard scaled-dot-product (no flash)
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    pad_id = tokenizer.pad_token_id or 0

    # ── Random inputs ─────────────────────────────────────────────────────────
    input_ids_list = [
        [randint(0, VOCAB_SIZE - 1) for _ in range(randint(50, MAX_INPUT))]
        for _ in range(NUM_SEQS)
    ]
    max_output_per_seq = [randint(50, MAX_OUTPUT) for _ in range(NUM_SEQS)]

    # ── Warmup ────────────────────────────────────────────────────────────────
    with nvtx_range("warmup"):
        dummy = torch.zeros(1, 32, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            model.generate(dummy, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ── Prepare batch (HF requires uniform length → padding) ─────────────────
    # This padding is a core inefficiency: tokens past each sequence's real end
    # still occupy KV cache slots.
    with nvtx_range("prepare_batch_with_padding"):
        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = [
            [pad_id] * (max_len - len(ids)) + ids
            for ids in input_ids_list
        ]
        attn_mask = [
            [0] * (max_len - len(ids)) + [1] * len(ids)
            for ids in input_ids_list
        ]
        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device="cuda")
        mask_tensor  = torch.tensor(attn_mask,  dtype=torch.long, device="cuda")

    # ── Generate ──────────────────────────────────────────────────────────────
    # All sequences run for max(max_output_per_seq) steps regardless of whether
    # each individual sequence has already produced enough tokens.
    max_new = max(max_output_per_seq)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with nvtx_range(f"hf_generate[batch={NUM_SEQS},max_new={max_new}]"):
        with torch.inference_mode():
            output = model.generate(
                input_tensor,
                attention_mask=mask_tensor,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,      # standard KV cache (contiguous, per-batch)
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    # Output tokens = actual new tokens (excl. input)
    total_out_toks = (output.shape[1] - input_tensor.shape[1]) * NUM_SEQS
    peak_mem_gb    = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*60}")
    print(f"BASELINE (HuggingFace, no paged attention)")
    print(f"  Sequences : {NUM_SEQS}")
    print(f"  Max input : {MAX_INPUT} tok  |  Max output: {MAX_OUTPUT} tok")
    print(f"  Total out : {total_out_toks} tok")
    print(f"  Time      : {elapsed:.2f}s")
    print(f"  Throughput: {total_out_toks / elapsed:.1f} tok/s")
    print(f"  Peak VRAM : {peak_mem_gb:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
