#!/usr/bin/env python3
"""Isolated warm-burst probe for bounded full-attention windows on Qwen3.5."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import batch_generate, generate_step
from mlx_lm.models.cache import KVCache, load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load


PREFIX_SEED = (
    "Apple Silicon runtimes need explicit bounded KV ownership. "
    "This synthetic prefix is intentionally repetitive to stress long-context "
    "cache residency on hybrid Qwen3.5 paths. "
)

SUFFIX_TEMPLATE = (
    "Request {idx}: answer briefly about bounded KV, prefix reuse, and serving latency. "
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


def bound_full_attention_layers(prompt_cache, window_tokens: int) -> int:
    trimmed = 0
    for cache in prompt_cache:
        if not isinstance(cache, KVCache) or cache.keys is None:
            continue
        if cache.offset <= window_tokens:
            continue
        cache.state = (
            mx.contiguous(cache.keys[..., -window_tokens:, :]),
            mx.contiguous(cache.values[..., -window_tokens:, :]),
        )
        trimmed += 1
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm burst probe for Qwen3.5 bounded FA windows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=None,
        help="If set, trim only full-attention KVCache layers to this window before the warm burst.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    prefix = build_exact_tokens(tokenizer, PREFIX_SEED, args.prefix_tokens)
    suffixes = [
        build_exact_tokens(tokenizer, SUFFIX_TEMPLATE.format(idx=i), args.suffix_tokens)
        for i in range(args.concurrency)
    ]

    base_cache = make_prompt_cache(model)
    for _ in generate_step(
        mx.array(prefix),
        model,
        max_tokens=0,
        prompt_cache=base_cache,
    ):
        pass

    with tempfile.TemporaryDirectory(prefix="siliconrt-qwen35-burst-") as tmpdir:
        cache_path = Path(tmpdir) / "shared_cache.safetensors"
        save_prompt_cache(str(cache_path), base_cache, metadata={"tokenizer_config": "{}"})
        prompt_caches = [load_prompt_cache(str(cache_path)) for _ in range(args.concurrency)]

        trimmed_layers = 0
        if args.window_tokens is not None:
            for cache in prompt_caches:
                trimmed_layers = bound_full_attention_layers(cache, args.window_tokens)

        response = batch_generate(
            model,
            tokenizer,
            prompts=suffixes,
            prompt_caches=prompt_caches,
            max_tokens=args.generation_tokens,
            verbose=False,
            return_prompt_caches=True,
            prefill_batch_size=args.concurrency,
            completion_batch_size=args.concurrency,
        )

    final_cache_nbytes = [cache_nbytes(c) for c in (response.caches or [])]
    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "generation_tokens": args.generation_tokens,
        "concurrency": args.concurrency,
        "window_tokens": args.window_tokens,
        "trimmed_full_attention_layers": trimmed_layers,
        "prompt_time_s": response.stats.prompt_time,
        "generation_time_s": response.stats.generation_time,
        "prompt_tps": response.stats.prompt_tps,
        "generation_tps": response.stats.generation_tps,
        "peak_memory_gb": response.stats.peak_memory,
        "avg_final_cache_nbytes": (
            sum(final_cache_nbytes) / len(final_cache_nbytes) if final_cache_nbytes else 0
        ),
        "final_cache_nbytes_per_request": final_cache_nbytes,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
