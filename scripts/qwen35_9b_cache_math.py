#!/usr/bin/env python3
"""CPU-only cache math for the local Qwen3.5-9B profile."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis import estimate_qwen35_9b_text_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B text-cache math.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    args = parser.parse_args()

    estimate = estimate_qwen35_9b_text_cache(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        bytes_per_element=args.dtype_bytes,
    )
    print(json.dumps(estimate.to_dict(), indent=2))


if __name__ == "__main__":
    main()
