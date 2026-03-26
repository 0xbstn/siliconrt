#!/usr/bin/env python3
"""Probe cold-vs-warm prefix behavior for mlx_lm.

This script is intentionally narrow. It measures the first benchmark that
matters for the KV-first roadmap:

- cold TTFT for (prefix + suffix)
- warm TTFT after caching the prefix once

It uses raw token sequences directly instead of chat templates so the result is
stable and easy to reason about.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load


PREFIX_SEED = (
    "Apple Silicon inference systems need careful runtime design. "
    "This prefix is intentionally repetitive so we can synthesize a long "
    "cached context without depending on product-specific prompts. "
)

SUFFIX_SEED = (
    "Given the cached context above, continue with a short technical answer "
    "about KV cache reuse and latency. "
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

    post_first_tps = None
    if generated > 1 and end > first_token_at:
        post_first_tps = (generated - 1) / (end - first_token_at)

    return {
        "ttft_s": first_token_at - start,
        "total_s": end - start,
        "generated_tokens": generated,
        "post_first_token_tps": post_first_tps,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def build_prefix_cache(prefix_tokens, model, *, kv_bits=None, kv_group_size=64):
    cache = make_prompt_cache(model)
    for _ in generate_step(
        mx.array(prefix_tokens),
        model,
        max_tokens=0,
        prompt_cache=cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    ):
        pass
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold vs warm prefix probe for mlx_lm.")
    parser.add_argument("--model", required=True, help="Local model path or repo id.")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)

    prefix_tokens = build_exact_tokens(tokenizer, PREFIX_SEED, args.prefix_tokens)
    suffix_tokens = build_exact_tokens(tokenizer, SUFFIX_SEED, args.suffix_tokens)
    full_prompt = prefix_tokens + suffix_tokens

    prefix_cache = build_prefix_cache(
        prefix_tokens,
        model,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
    )

    with tempfile.TemporaryDirectory(prefix="siliconrt-kv-probe-") as tmpdir:
        cache_path = Path(tmpdir) / "prefix_cache.safetensors"
        save_prompt_cache(
            str(cache_path),
            prefix_cache,
            metadata={"model": args.model, "tokenizer_config": "{}"},
        )

        cold_trials = []
        warm_trials = []

        for _ in range(args.trials):
            mx.clear_cache()
            mx.reset_peak_memory()
            cold_trials.append(
                measure_generation(
                    full_prompt,
                    model,
                    max_tokens=args.generation_tokens,
                    prompt_cache=None,
                )
            )

        for _ in range(args.trials):
            warm_cache = load_prompt_cache(str(cache_path))
            mx.clear_cache()
            mx.reset_peak_memory()
            warm_trials.append(
                measure_generation(
                    suffix_tokens,
                    model,
                    max_tokens=args.generation_tokens,
                    prompt_cache=warm_cache,
                )
            )

    def avg(key: str, rows: list[dict]) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    summary = {
        "model": args.model,
        "prefix_tokens": len(prefix_tokens),
        "suffix_tokens": len(suffix_tokens),
        "generation_tokens": args.generation_tokens,
        "trials": args.trials,
        "cold_avg_ttft_s": avg("ttft_s", cold_trials),
        "warm_avg_ttft_s": avg("ttft_s", warm_trials),
        "cold_avg_post_first_tps": avg("post_first_token_tps", cold_trials),
        "warm_avg_post_first_tps": avg("post_first_token_tps", warm_trials),
        "cold_avg_peak_memory_gb": avg("peak_memory_gb", cold_trials),
        "warm_avg_peak_memory_gb": avg("peak_memory_gb", warm_trials),
        "speedup_ttft": (
            avg("ttft_s", cold_trials) / avg("ttft_s", warm_trials)
            if avg("ttft_s", cold_trials) and avg("ttft_s", warm_trials)
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
