#!/usr/bin/env python3
"""Long-context recall probe for bounded Qwen3.5 full-attention windows."""

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


FILLER_SEED = (
    "This archive contains repeated engineering notes about Apple Silicon runtimes, "
    "bounded KV windows, shared-prefix serving, and hybrid attention models. "
)


def build_exact_tokens(tokenizer, seed: str, target_len: int) -> list[int]:
    chunk = tokenizer.encode(seed)
    if not chunk:
        raise ValueError("Tokenizer produced an empty chunk.")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(chunk)
    return out[:target_len]


def clone_prompt_cache(cache) -> list:
    with tempfile.TemporaryDirectory(prefix="siliconrt-qwen35-recall-") as tmpdir:
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


def normalize_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def build_case_prefix(
    tokenizer,
    *,
    prefix_tokens: int,
    distance_tokens: int,
    secret: str,
) -> list[int]:
    secret_line = (
        f"\nARCHIVE RECORD: the unique passcode is {secret}. "
        "Memorize the exact passcode.\n"
    )
    secret_tokens = tokenizer.encode(secret_line)
    if len(secret_tokens) >= prefix_tokens:
        raise ValueError("Secret line is longer than the target prefix.")
    after_len = max(0, distance_tokens - len(secret_tokens))
    before_len = prefix_tokens - len(secret_tokens) - after_len
    filler_before = build_exact_tokens(tokenizer, FILLER_SEED, before_len)
    filler_after = build_exact_tokens(tokenizer, FILLER_SEED, after_len)
    return filler_before + secret_tokens + filler_after


def build_question_tokens(tokenizer) -> list[int]:
    prompt = (
        "Question: what is the unique passcode from the archive record? "
        "Reply with only the passcode.\nAnswer:"
    )
    return tokenizer.encode(prompt)


def generate_answer(question_tokens, model, *, prompt_cache, max_tokens: int, tokenizer) -> dict:
    start = time.perf_counter()
    tokens: list[int] = []
    for token, _logprobs in generate_step(
        mx.array(question_tokens),
        model,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
    ):
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        tokens.append(token_id)
    end = time.perf_counter()
    text = tokenizer.decode(tokens)
    return {
        "answer_text": text,
        "answer_tokens": tokens,
        "total_s": end - start,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Qwen3.5 long-context recall probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--distance-tokens", type=int, action="append", required=True)
    parser.add_argument("--window-tokens", type=int, action="append", default=[])
    parser.add_argument("--generation-tokens", type=int, default=12)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    question_tokens = build_question_tokens(tokenizer)
    window_values = [None] + list(args.window_tokens)
    rows: list[dict] = []

    for case_idx, distance_tokens in enumerate(args.distance_tokens, start=1):
        secret = f"MAC-{case_idx:02d}-WINDOW-{distance_tokens}"
        prefix = build_case_prefix(
            tokenizer,
            prefix_tokens=args.prefix_tokens,
            distance_tokens=distance_tokens,
            secret=secret,
        )
        base_cache = make_prompt_cache(model)
        for _token, _logprobs in generate_step(
            mx.array(prefix),
            model,
            max_tokens=0,
            prompt_cache=base_cache,
        ):
            pass

        for window_tokens in window_values:
            prompt_cache = clone_prompt_cache(base_cache)
            trimmed_layers = 0
            if window_tokens is not None:
                trimmed_layers = bound_full_attention_layers(prompt_cache, window_tokens)

            mx.clear_cache()
            mx.reset_peak_memory()
            answer = generate_answer(
                question_tokens,
                model,
                prompt_cache=prompt_cache,
                max_tokens=args.generation_tokens,
                tokenizer=tokenizer,
            )
            normalized_answer = normalize_text(answer["answer_text"])
            normalized_secret = normalize_text(secret)
            rows.append(
                {
                    "distance_tokens": distance_tokens,
                    "window_tokens": window_tokens,
                    "trimmed_full_attention_layers": trimmed_layers,
                    "secret": secret,
                    "answer_text": answer["answer_text"],
                    "normalized_answer": normalized_answer,
                    "normalized_secret": normalized_secret,
                    "starts_with_secret": normalized_answer.startswith(normalized_secret),
                    "contains_secret": normalized_secret in normalized_answer,
                    "total_s": answer["total_s"],
                    "peak_memory_gb": answer["peak_memory_gb"],
                }
            )

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "generation_tokens": args.generation_tokens,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
