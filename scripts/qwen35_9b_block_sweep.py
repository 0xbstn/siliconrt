#!/usr/bin/env python3
"""CPU-only block-size sweep for the local Qwen3.5-9B cache profile."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis import estimate_qwen35_9b_text_cache
from siliconrt.analysis.paged_kv_math import (
    block_utilization,
    estimate_paged_kv,
    max_sessions_paged_under_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B paged-KV block sweep.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--budget-gb", type=float, default=None)
    parser.add_argument("--block-tokens", type=int, action="append", required=True)
    parser.add_argument("--block-table-bytes", type=int, default=16)
    args = parser.parse_args()

    profile = estimate_qwen35_9b_text_cache(seq_len=args.seq_len)
    rows = []
    for block_tokens in args.block_tokens:
        paged = estimate_paged_kv(
            seq_len=args.seq_len,
            block_tokens=block_tokens,
            bytes_per_token=profile.full_attention_bytes_per_token_total,
            constant_bytes=profile.linear_attention_total_bytes,
            block_table_bytes_per_block=args.block_table_bytes,
        )
        row = {
            "block_tokens": block_tokens,
            "utilization": block_utilization(args.seq_len, block_tokens),
            "num_blocks": paged.num_blocks,
            "payload_bytes": paged.payload_bytes,
            "reserved_payload_bytes": paged.reserved_payload_bytes,
            "wasted_payload_bytes": paged.wasted_payload_bytes,
            "block_table_bytes": paged.block_table_bytes,
            "total_bytes": paged.total_bytes,
            "total_gb": paged.total_gb,
        }
        if args.budget_gb is not None:
            row["max_sessions_under_budget"] = max_sessions_paged_under_budget(
                budget_bytes=args.budget_gb * 1e9,
                seq_len=args.seq_len,
                block_tokens=block_tokens,
                bytes_per_token=profile.full_attention_bytes_per_token_total,
                constant_bytes=profile.linear_attention_total_bytes,
                block_table_bytes_per_block=args.block_table_bytes,
            )
        rows.append(row)

    out = {
        "seq_len": args.seq_len,
        "full_attention_bytes_per_token_total": profile.full_attention_bytes_per_token_total,
        "constant_bytes": profile.linear_attention_total_bytes,
        "block_table_bytes_per_block": args.block_table_bytes,
        "rows": rows,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
