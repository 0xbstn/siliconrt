#!/usr/bin/env python3
"""Prototype a bounded full-attention cache path for Qwen3.5.

This script keeps the linear-attention ArraysCache state intact and manually
rebases only the full-attention KVCache layers to the last N tokens.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load


PREFIX_SEED = (
    "Apple Silicon runtimes need explicit bounded KV ownership. "
    "This synthetic prefix is intentionally repetitive to stress long-context "
    "cache residency on hybrid Qwen3.5 paths. "
)

SUFFIX_SEED = (
    "Continue with a short technical answer about bounded KV windows and "
    "hybrid attention caches. "
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


def clone_prompt_cache(cache) -> list:
    with tempfile.TemporaryDirectory(prefix="siliconrt-qwen35-clone-") as tmpdir:
        path = Path(tmpdir) / "cache.safetensors"
        save_prompt_cache(str(path), cache, metadata={"tokenizer_config": "{}"})
        return load_prompt_cache(str(path))


def bound_full_attention_layers(prompt_cache, window_tokens: int) -> int:
    trimmed_layers = 0
    for cache in prompt_cache:
        if not isinstance(cache, KVCache) or cache.keys is None:
            continue
        if cache.offset <= window_tokens:
            continue
        cache.state = (
            mx.contiguous(cache.keys[..., -window_tokens:, :]),
            mx.contiguous(cache.values[..., -window_tokens:, :]),
        )
        trimmed_layers += 1
    return trimmed_layers


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
        "final_cache_nbytes": cache_nbytes(prompt_cache) if prompt_cache is not None else 0,
    }


def measure_generation_with_bounded_decode(
    prompt_tokens,
    model,
    *,
    max_tokens: int,
    prompt_cache,
    window_tokens: int,
) -> dict:
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
        bound_full_attention_layers(prompt_cache, window_tokens)

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
        "final_cache_nbytes": cache_nbytes(prompt_cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bound full-attention cache probe for Qwen3.5.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--window-tokens", type=int, default=4096)
    parser.add_argument(
        "--maintain-window-during-decode",
        action="store_true",
        help="Re-apply the full-attention window after each generated token.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    prefix_tokens = build_exact_tokens(tokenizer, PREFIX_SEED, args.prefix_tokens)
    suffix_tokens = build_exact_tokens(tokenizer, SUFFIX_SEED, args.suffix_tokens)

    base_cache = make_prompt_cache(model)
    for _ in generate_step(
        mx.array(prefix_tokens),
        model,
        max_tokens=0,
        prompt_cache=base_cache,
    ):
        pass

    unbounded_cache = clone_prompt_cache(base_cache)
    bounded_cache = clone_prompt_cache(base_cache)

    unbounded_nbytes = cache_nbytes(unbounded_cache)
    trimmed_layers = bound_full_attention_layers(bounded_cache, args.window_tokens)
    bounded_nbytes = cache_nbytes(bounded_cache)

    mx.clear_cache()
    mx.reset_peak_memory()
    unbounded_metrics = measure_generation(
        suffix_tokens,
        model,
        max_tokens=args.generation_tokens,
        prompt_cache=unbounded_cache,
    )

    mx.clear_cache()
    mx.reset_peak_memory()
    if args.maintain_window_during_decode:
        bounded_metrics = measure_generation_with_bounded_decode(
            suffix_tokens,
            model,
            max_tokens=args.generation_tokens,
            prompt_cache=bounded_cache,
            window_tokens=args.window_tokens,
        )
    else:
        bounded_metrics = measure_generation(
            suffix_tokens,
            model,
            max_tokens=args.generation_tokens,
            prompt_cache=bounded_cache,
        )

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "generation_tokens": args.generation_tokens,
        "window_tokens": args.window_tokens,
        "trimmed_full_attention_layers": trimmed_layers,
        "unbounded_cache_nbytes": unbounded_nbytes,
        "bounded_cache_nbytes": bounded_nbytes,
        "cache_nbytes_reduction": unbounded_nbytes - bounded_nbytes,
        "cache_nbytes_reduction_ratio": (
            (unbounded_nbytes - bounded_nbytes) / unbounded_nbytes
            if unbounded_nbytes
            else 0
        ),
        "maintain_window_during_decode": args.maintain_window_during_decode,
        "unbounded_metrics": unbounded_metrics,
        "bounded_metrics": bounded_metrics,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
