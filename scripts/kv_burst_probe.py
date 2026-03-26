#!/usr/bin/env python3
"""Shared-prefix burst probe for mlx_lm.

This measures a serving-style batch where several requests share the same long
prefix. We compare:

- cold batch: each request sends prefix + suffix
- warm batch: prefix is pre-cached and each request sends only its suffix
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import batch_generate, generate_step
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load


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


def build_prefix_cache(prefix_tokens, model):
    cache = make_prompt_cache(model)
    for _ in generate_step(
        mx.array(prefix_tokens),
        model,
        max_tokens=0,
        prompt_cache=cache,
    ):
        pass
    return cache


def run_batch(model, tokenizer, prompts, *, prompt_caches, generation_tokens: int):
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
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
    end = time.perf_counter()
    return {
        "wall_s": end - start,
        "prompt_time_s": response.stats.prompt_time,
        "generation_time_s": response.stats.generation_time,
        "prompt_tps": response.stats.prompt_tps,
        "generation_tps": response.stats.generation_tps,
        "peak_memory_gb": response.stats.peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-prefix burst probe for mlx_lm.")
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

    prefix_cache = build_prefix_cache(prefix_tokens, model)

    with tempfile.TemporaryDirectory(prefix="siliconrt-kv-burst-") as tmpdir:
        cache_path = Path(tmpdir) / "shared_prefix_cache.safetensors"
        save_prompt_cache(
            str(cache_path),
            prefix_cache,
            metadata={"model": args.model, "tokenizer_config": "{}"},
        )

        cold_trials = []
        warm_trials = []

        for _ in range(args.trials):
            cold_trials.append(
                run_batch(
                    model,
                    tokenizer,
                    cold_prompts,
                    prompt_caches=None,
                    generation_tokens=args.generation_tokens,
                )
            )

        for _ in range(args.trials):
            warm_caches = [
                load_prompt_cache(str(cache_path)) for _ in range(args.concurrency)
            ]
            warm_trials.append(
                run_batch(
                    model,
                    tokenizer,
                    suffix_prompts,
                    prompt_caches=warm_caches,
                    generation_tokens=args.generation_tokens,
                )
            )

    def avg(key: str, rows: list[dict]) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "generation_tokens": args.generation_tokens,
        "concurrency": args.concurrency,
        "trials": args.trials,
        "cold_avg_wall_s": avg("wall_s", cold_trials),
        "warm_avg_wall_s": avg("wall_s", warm_trials),
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
        "cold_trials": cold_trials,
        "warm_trials": warm_trials,
    }

    print(json.dumps(summary, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
