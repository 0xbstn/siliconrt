#!/usr/bin/env python3
"""CPU-only partitioned sequence/constant sizing for the local Qwen3.5-9B profile."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis.partitioned_runtime_math import (
    estimate_qwen35_partitioned_session,
    make_sequence_biased_qwen35_plan,
    max_partitioned_sessions_under_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B partitioned runtime sizing."
    )
    parser.add_argument("--budget-gb", type=float, action="append", required=True)
    parser.add_argument("--window-len", type=int, action="append", required=True)
    parser.add_argument("--target-sessions", type=int, action="append", default=[])
    args = parser.parse_args()

    rows = []
    for budget_gb in args.budget_gb:
        budget_bytes = int(budget_gb * 1e9)
        for window_len in args.window_len:
            session = estimate_qwen35_partitioned_session(window_len=window_len)
            max_sessions = max_partitioned_sessions_under_budget(
                budget_bytes=budget_bytes,
                window_len=window_len,
            )
            targets = list(args.target_sessions) or [max_sessions]
            for target_sessions in targets:
                rows.append(
                    make_sequence_biased_qwen35_plan(
                        budget_bytes=budget_bytes,
                        window_len=window_len,
                        target_sessions=target_sessions,
                    ).to_dict()
                    | {
                        "budget_gb": budget_gb,
                        "max_sessions_theoretical": max_sessions,
                        "per_session": session.to_dict(),
                    }
                )

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
