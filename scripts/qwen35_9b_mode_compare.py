#!/usr/bin/env python3
"""CPU-only memory-mode comparison for the local Qwen3.5-9B profile."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis.workload_mix import simulate_memory_mode_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B memory-mode comparison.")
    parser.add_argument("--budget-gb", type=float, required=True)
    parser.add_argument("--seq-len", type=int, action="append", required=True)
    parser.add_argument("--window-len", type=int, action="append", default=[])
    parser.add_argument("--block-tokens", type=int, action="append", default=[])
    parser.add_argument("--evict-oldest", action="store_true")
    parser.add_argument("--block-table-bytes", type=int, default=16)
    args = parser.parse_args()

    budget_bytes = int(args.budget_gb * 1e9)
    rows = []

    rows.append(
        simulate_memory_mode_workload(
            budget_bytes=budget_bytes,
            seq_lens=list(args.seq_len),
            mode="contiguous_unbounded",
            evict_oldest=args.evict_oldest,
        )
    )

    for window_len in args.window_len:
        rows.append(
            simulate_memory_mode_workload(
                budget_bytes=budget_bytes,
                seq_lens=list(args.seq_len),
                mode="bounded_contiguous",
                window_len=window_len,
                evict_oldest=args.evict_oldest,
            )
        )

    for block_tokens in args.block_tokens:
        rows.append(
            simulate_memory_mode_workload(
                budget_bytes=budget_bytes,
                seq_lens=list(args.seq_len),
                mode="paged",
                block_tokens=block_tokens,
                block_table_bytes_per_block=args.block_table_bytes,
                evict_oldest=args.evict_oldest,
            )
        )

    print(json.dumps({"budget_bytes": budget_bytes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
