#!/usr/bin/env python3
"""Probe siliconrt in-memory prefix snapshot restore behavior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.utils import load

from siliconrt.prefix import PrefixStore


PREFIX_SEED = (
    "Apple Silicon serving workloads often repeat a large shared system prompt. "
    "This synthetic prefix is intentionally repetitive so we can stress prompt reuse. "
)

SUFFIX_SEED = (
    "Continue with a short answer about cache reuse, latency, and memory behavior. "
)


def build_exact_tokens(tokenizer, seed: str, target_len: int) -> list[int]:
    chunk = tokenizer.encode(seed)
    if not chunk:
        raise ValueError("Tokenizer produced an empty chunk.")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(chunk)
    return out[:target_len]


def measure_generation(prompt_tokens, model, *, max_tokens: int, prompt_cache=None) -> dict:
    start = time.perf_counter()
    first_token_at = None
    generated = 0

    for _token, _logprobs in generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
    ):
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        generated += 1

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("Generation produced no tokens.")

    return {
        "ttft_s": first_token_at - start,
        "total_s": end - start,
        "generated_tokens": generated,
        "post_first_token_tps": (
            (generated - 1) / (end - first_token_at)
            if generated > 1 and end > first_token_at
            else None
        ),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="siliconrt in-memory prefix probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    prefix_tokens = build_exact_tokens(tokenizer, PREFIX_SEED, args.prefix_tokens)
    suffix_tokens = build_exact_tokens(tokenizer, SUFFIX_SEED, args.suffix_tokens)

    store = PrefixStore()
    snapshot = store.capture(
        model=model,
        model_key=args.model,
        prefix_tokens=prefix_tokens,
    )

    rows = []
    for _ in range(args.trials):
        mx.clear_cache()
        mx.reset_peak_memory()
        restore_start = time.perf_counter()
        prompt_cache = store.checkout(snapshot.key, materialize=True)
        restore_end = time.perf_counter()
        generation = measure_generation(
            suffix_tokens,
            model,
            max_tokens=args.generation_tokens,
            prompt_cache=prompt_cache,
        )
        rows.append(
            {
                "restore_s": restore_end - restore_start,
                "ttft_s": generation["ttft_s"],
                "total_s": generation["total_s"],
                "generated_tokens": generation["generated_tokens"],
                "post_first_token_tps": generation["post_first_token_tps"],
                "peak_memory_gb": generation["peak_memory_gb"],
            }
        )

    def avg(key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "generation_tokens": args.generation_tokens,
        "trials": args.trials,
        "snapshot_nbytes": snapshot.nbytes,
        "avg_restore_s": avg("restore_s"),
        "avg_ttft_s": avg("ttft_s"),
        "avg_post_first_token_tps": avg("post_first_token_tps"),
        "avg_peak_memory_gb": avg("peak_memory_gb"),
        "trials_data": rows,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
