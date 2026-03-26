#!/usr/bin/env python3
"""CPU-only KV memory calculator for siliconrt."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis import (
    bounded_kv_savings_factor,
    estimate_kv_memory,
    max_sessions_under_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only KV memory math.")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--budget-gb", type=float, default=None)
    parser.add_argument("--window-len", type=int, default=None)
    args = parser.parse_args()

    estimate = estimate_kv_memory(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    out = {
        "estimate": estimate.to_dict(),
    }

    if args.budget_gb is not None:
        out["budget_bytes"] = args.budget_gb * 1e9
        out["max_sessions_under_budget"] = max_sessions_under_budget(
            budget_bytes=out["budget_bytes"],
            layers=args.layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            seq_len=args.seq_len,
            dtype=args.dtype,
        )

    if args.window_len is not None:
        out["window_len"] = args.window_len
        out["bounded_kv_savings_factor"] = bounded_kv_savings_factor(
            seq_len=args.seq_len,
            window_len=args.window_len,
        )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
