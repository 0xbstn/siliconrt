#!/usr/bin/env python3
"""Shared-prefix burst probe for siliconrt's in-memory prefix store."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import batch_generate
from mlx_lm.utils import load

from siliconrt.prefix import PrefixStore


PREFIX_SEED = (
    "Apple Silicon serving workloads often repeat a large shared system prompt. "
    "This synthetic prefix is intentionally repetitive so we can stress prompt reuse. "
)

SUFFIX_TEMPLATE = (
    "Request {idx}: answer briefly about cache reuse, latency, and memory behavior. "
)


def build_exact_tokens(tokenizer, seed: str, target_len: int) -> list[int]:
    chunk = tokenizer.encode(seed)
    if not chunk:
        raise ValueError("Tokenizer produced an empty chunk.")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(chunk)
    return out[:target_len]


def run_cold_batch(model, tokenizer, prompts, *, generation_tokens: int) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    response = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        prompt_caches=None,
        max_tokens=generation_tokens,
        verbose=False,
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
    )
    end = time.perf_counter()
    return {
        "wall_s": end - start,
        "prompt_time_s": response.stats.prompt_time,
        "generation_time_s": response.stats.generation_time,
        "prompt_tps": response.stats.prompt_tps,
        "generation_tps": response.stats.generation_tps,
        "peak_memory_gb": response.stats.peak_memory,
    }


def run_warm_batch(
    model,
    tokenizer,
    prompts,
    *,
    store: PrefixStore,
    snapshot_key: str,
    concurrency: int,
    generation_tokens: int,
) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    restore_start = time.perf_counter()
    prompt_caches = store.get(snapshot_key).restore_many(concurrency, materialize=True)
    restore_end = time.perf_counter()
    batch_start = time.perf_counter()
    response = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        prompt_caches=prompt_caches,
        max_tokens=generation_tokens,
        verbose=False,
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
    )
    batch_end = time.perf_counter()
    return {
        "restore_s": restore_end - restore_start,
        "wall_s": batch_end - batch_start,
        "wall_plus_restore_s": batch_end - restore_start,
        "prompt_time_s": response.stats.prompt_time,
        "generation_time_s": response.stats.generation_time,
        "prompt_tps": response.stats.prompt_tps,
        "generation_tps": response.stats.generation_tps,
        "peak_memory_gb": response.stats.peak_memory,
    }


def avg(key: str, rows: list[dict]) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-prefix burst probe for siliconrt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)

    prefix_tokens = build_exact_tokens(tokenizer, PREFIX_SEED, args.prefix_tokens)
    suffix_prompts = [
        build_exact_tokens(
            tokenizer,
            SUFFIX_TEMPLATE.format(idx=i),
            args.suffix_tokens,
        )
        for i in range(args.concurrency)
    ]
    cold_prompts = [prefix_tokens + suffix for suffix in suffix_prompts]

    store = PrefixStore()
    snapshot = store.capture(
        model=model,
        model_key=args.model,
        prefix_tokens=prefix_tokens,
    )

    cold_trials = []
    warm_trials = []

    for _ in range(args.trials):
        cold_trials.append(
            run_cold_batch(
                model,
                tokenizer,
                cold_prompts,
                generation_tokens=args.generation_tokens,
            )
        )

    for _ in range(args.trials):
        warm_trials.append(
            run_warm_batch(
                model,
                tokenizer,
                suffix_prompts,
                store=store,
                snapshot_key=snapshot.key,
                concurrency=args.concurrency,
                generation_tokens=args.generation_tokens,
            )
        )

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "generation_tokens": args.generation_tokens,
        "concurrency": args.concurrency,
        "trials": args.trials,
        "snapshot_nbytes": snapshot.nbytes,
        "cold_avg_wall_s": avg("wall_s", cold_trials),
        "warm_avg_restore_s": avg("restore_s", warm_trials),
        "warm_avg_wall_s": avg("wall_s", warm_trials),
        "warm_avg_wall_plus_restore_s": avg("wall_plus_restore_s", warm_trials),
        "cold_avg_prompt_time_s": avg("prompt_time_s", cold_trials),
        "warm_avg_prompt_time_s": avg("prompt_time_s", warm_trials),
        "cold_avg_generation_time_s": avg("generation_time_s", cold_trials),
        "warm_avg_generation_time_s": avg("generation_time_s", warm_trials),
        "cold_avg_prompt_tps": avg("prompt_tps", cold_trials),
        "warm_avg_prompt_tps": avg("prompt_tps", warm_trials),
        "cold_avg_generation_tps": avg("generation_tps", cold_trials),
        "warm_avg_generation_tps": avg("generation_tps", warm_trials),
        "cold_avg_peak_memory_gb": avg("peak_memory_gb", cold_trials),
        "warm_avg_peak_memory_gb": avg("peak_memory_gb", warm_trials),
        "wall_speedup": (
            avg("wall_s", cold_trials) / avg("wall_s", warm_trials)
            if avg("wall_s", cold_trials) and avg("wall_s", warm_trials)
            else None
        ),
        "wall_plus_restore_speedup": (
            avg("wall_s", cold_trials) / avg("wall_plus_restore_s", warm_trials)
            if avg("wall_s", cold_trials) and avg("wall_plus_restore_s", warm_trials)
            else None
        ),
        "cold_trials": cold_trials,
        "warm_trials": warm_trials,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
