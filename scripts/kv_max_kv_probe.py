#!/usr/bin/env python3
"""Probe whether mlx_lm's max_kv_size materially affects a workload.

This is intentionally narrow. It targets one concrete question:

- on models with a custom make_cache(), does `max_kv_size` actually reduce
  prompt-cache residency or prompt-time behavior?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm.generate import batch_generate
from mlx_lm.utils import load


PREFIX_SEED = (
    "Apple Silicon runtimes need explicit cache ownership, bounded memory, "
    "and measurable runtime seams. This synthetic prefix is intentionally "
    "repetitive to stress KV behavior. "
)

SUFFIX_TEMPLATE = (
    "Request {idx}: answer briefly about bounded KV, cache pressure, and "
    "prefix reuse. "
)


def build_exact_tokens(tokenizer, seed: str, target_len: int) -> list[int]:
    chunk = tokenizer.encode(seed)
    if not chunk:
        raise ValueError("Tokenizer produced an empty chunk.")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(chunk)
    return out[:target_len]


def cache_nbytes(prompt_cache) -> int:
    return sum(getattr(c, "nbytes", 0) for c in prompt_cache)


def run_case(
    model,
    tokenizer,
    *,
    prefix_tokens: int,
    suffix_tokens: int,
    generation_tokens: int,
    concurrency: int,
    max_kv_size: int | None,
    prefill_step_size: int,
) -> dict:
    prefix = build_exact_tokens(tokenizer, PREFIX_SEED, prefix_tokens)
    prompts = [
        prefix
        + build_exact_tokens(tokenizer, SUFFIX_TEMPLATE.format(idx=i), suffix_tokens)
        for i in range(concurrency)
    ]

    response = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        max_tokens=generation_tokens,
        verbose=False,
        return_prompt_caches=True,
        prefill_batch_size=concurrency,
        completion_batch_size=concurrency,
        prefill_step_size=prefill_step_size,
        max_kv_size=max_kv_size,
    )

    prompt_cache_nbytes = [cache_nbytes(c) for c in (response.caches or [])]
    cache_type_counts = {}
    if response.caches:
        for cache in response.caches[0]:
            name = type(cache).__name__
            cache_type_counts[name] = cache_type_counts.get(name, 0) + 1

    return {
        "max_kv_size": max_kv_size,
        "prefill_step_size": prefill_step_size,
        "concurrency": concurrency,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "generation_tokens": generation_tokens,
        "prompt_time_s": response.stats.prompt_time,
        "generation_time_s": response.stats.generation_time,
        "prompt_tps": response.stats.prompt_tps,
        "generation_tps": response.stats.generation_tps,
        "peak_memory_gb": response.stats.peak_memory,
        "prompt_tokens": response.stats.prompt_tokens,
        "generated_tokens": response.stats.generation_tokens,
        "prompt_cache_nbytes_per_request": prompt_cache_nbytes,
        "avg_prompt_cache_nbytes": (
            sum(prompt_cache_nbytes) / len(prompt_cache_nbytes)
            if prompt_cache_nbytes
            else 0
        ),
        "cache_type_counts": cache_type_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe max_kv_size behavior for mlx_lm.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--max-kv-sizes",
        type=str,
        default="none,4096,2048",
        help="Comma-separated values, use 'none' for no limit.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)

    max_kv_sizes: list[int | None] = []
    for raw in args.max_kv_sizes.split(","):
        raw = raw.strip().lower()
        if raw == "none":
            max_kv_sizes.append(None)
        else:
            max_kv_sizes.append(int(raw))

    rows = [
        run_case(
            model,
            tokenizer,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            generation_tokens=args.generation_tokens,
            concurrency=args.concurrency,
            max_kv_size=max_kv_size,
            prefill_step_size=args.prefill_step_size,
        )
        for max_kv_size in max_kv_sizes
    ]

    summary = {
        "model": args.model,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
