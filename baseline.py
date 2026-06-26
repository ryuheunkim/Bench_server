"""
Baseline: HuggingFace Transformers — standard (non-paged) KV cache.

Memory model:
  - All sequences padded to the same length → attention matrix [B, H, L, L].
  - KV cache is a single contiguous tensor for the whole batch; no block sharing.
  - No prefix caching, no on-demand allocation.

Compare with nanovllm_bench.py to see the effect of paged attention.
"""

import os
import time
import torch
from random import randint, seed
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")
NUM_SEQS    = 32    # RTX4050 6GB limit: attention matrix [B,H,L,L] grows with B
MAX_INPUT   = 128   # keep L small so [16,16,192,192] fp32 ≈ 144MB stays safe
MAX_OUTPUT  = 64
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
        [randint(0, VOCAB_SIZE - 1) for _ in range(randint(32, MAX_INPUT))]
        for _ in range(NUM_SEQS)
    ]
    max_output_per_seq = [randint(32, MAX_OUTPUT) for _ in range(NUM_SEQS)]

    # ── Warmup ────────────────────────────────────────────────────────────────
    with nvtx_range("warmup"):
        dummy = torch.zeros(1, 32, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            model.generate(dummy, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ── Prepare padded batch ──────────────────────────────────────────────────
    # HF requires uniform sequence length → left-pad to max.
    # This is the core inefficiency being compared: padding wastes KV slots,
    # and the attention matrix is [B, H, L_max, L_max] for all sequences.
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

    # ── Generate (full batch) ─────────────────────────────────────────────────
    max_new = max(max_output_per_seq)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with nvtx_range("benchmark_generate"):
        with torch.inference_mode():
            with nvtx_range(f"hf_generate[batch={NUM_SEQS},max_new={max_new}]"):
                output = model.generate(
                    input_tensor,
                    attention_mask=mask_tensor,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

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
