#!/usr/bin/env python3
"""Bounded-memory pressure probe for siliconrt's prefix store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm.utils import load

from siliconrt.prefix import PrefixStore


PREFIX_TEMPLATE = (
    "Shared system prompt variant {idx}. "
    "Apple Silicon inference needs careful cache ownership and bounded memory behavior. "
)


def build_exact_tokens(tokenizer, seed: str, target_len: int) -> list[int]:
    chunk = tokenizer.encode(seed)
    if not chunk:
        raise ValueError("Tokenizer produced an empty chunk.")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(chunk)
    return out[:target_len]


def make_distinct_prefixes(tokenizer, *, prefix_tokens: int, count: int) -> list[list[int]]:
    return [
        build_exact_tokens(tokenizer, PREFIX_TEMPLATE.format(idx=i), prefix_tokens)
        for i in range(count)
    ]


def parse_byte_budget(args) -> int | None:
    if args.byte_budget_bytes is not None:
        return args.byte_budget_bytes
    if args.byte_budget_mb is not None:
        return int(args.byte_budget_mb * 1_000_000)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded-memory pressure probe for siliconrt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--byte-budget-mb", type=float, default=None)
    parser.add_argument("--byte-budget-bytes", type=int, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    byte_budget = parse_byte_budget(args)
    model, tokenizer = load(args.model)
    prefixes = make_distinct_prefixes(
        tokenizer,
        prefix_tokens=args.prefix_tokens,
        count=args.num_prefixes,
    )

    store = PrefixStore(byte_budget=byte_budget)
    rows = []

    for idx, prefix_tokens in enumerate(prefixes):
        snapshot = store.capture(
            model=model,
            model_key=args.model,
            prefix_tokens=prefix_tokens,
        )
        resident = store.contains(snapshot.key)
        rows.append(
            {
                "prefix_idx": idx,
                "snapshot_key": snapshot.key,
                "snapshot_nbytes": snapshot.nbytes,
                "resident_after_put": resident,
                "entries_after_put": len(store),
                "resident_bytes_after_put": store.nbytes,
                "evictions_after_put": store.evictions,
            }
        )

    summary = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "num_prefixes": args.num_prefixes,
        "byte_budget": byte_budget,
        "final_stats": store.stats(),
        "resident_keys": store.keys(),
        "max_snapshot_nbytes": max((row["snapshot_nbytes"] for row in rows), default=0),
        "rows": rows,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
