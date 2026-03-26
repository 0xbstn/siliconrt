#!/usr/bin/env python3
"""CPU-only admission and eviction simulation for Qwen3.5-9B cache handles."""

from __future__ import annotations

import argparse
from pathlib import Path

from siliconrt.analysis.arena_sim import (
    simulate_qwen35_prefix_admission,
    simulation_to_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B arena simulation.")
    parser.add_argument("--budget-gb", type=float, required=True)
    parser.add_argument("--seq-len", type=int, action="append", required=True)
    parser.add_argument("--evict-oldest", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    simulation = simulate_qwen35_prefix_admission(
        budget_bytes=int(args.budget_gb * 1e9),
        seq_lens=list(args.seq_len),
        evict_oldest=args.evict_oldest,
    )
    payload = simulation_to_json(simulation)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
