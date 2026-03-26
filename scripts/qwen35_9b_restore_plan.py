#!/usr/bin/env python3
"""CPU-only restore aliasing math for the local Qwen3.5-9B profile."""

from __future__ import annotations

import argparse
import json

from siliconrt.analysis.restore_math import (
    RestoreAliasMode,
    estimate_qwen35_restore_plan,
    estimate_qwen35_shared_prefix_runtime,
    estimate_qwen35_shared_prefix_mixed_promotion,
    max_qwen35_shared_prefix_concurrency_under_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B restore aliasing math.")
    parser.add_argument("--window-len", type=int, action="append", required=True)
    parser.add_argument("--budget-gb", type=float, action="append", default=[])
    parser.add_argument(
        "--concurrency",
        type=int,
        action="append",
        default=[],
        help="Concurrent decode restores sharing one prefix handle.",
    )
    parser.add_argument(
        "--promoted",
        type=int,
        action="append",
        default=[],
        help="Number of promoted decodes for copy-on-grow mixed-state estimates.",
    )
    args = parser.parse_args()

    rows = []
    for window_len in args.window_len:
        for mode in (
            RestoreAliasMode.CLONE_ALL,
            RestoreAliasMode.SHARE_CONSTANT_STATE,
            RestoreAliasMode.BORROW_SEQUENCE_AND_CONSTANT,
        ):
            rows.append(
                {
                    "kind": "restore_plan",
                    **estimate_qwen35_restore_plan(
                        window_len=window_len,
                        mode=mode,
                    ).to_dict(),
                }
            )
            for budget_gb in args.budget_gb:
                budget_bytes = int(budget_gb * 1e9)
                rows.append(
                    {
                        "kind": "budget_capacity",
                        "budget_gb": budget_gb,
                        "window_len": window_len,
                        "mode": str(mode),
                        "max_concurrent_decodes": max_qwen35_shared_prefix_concurrency_under_budget(
                            budget_bytes=budget_bytes,
                            window_len=window_len,
                            mode=mode,
                        ),
                    }
                )
            for concurrency in args.concurrency:
                rows.append(
                    {
                        "kind": "shared_prefix_runtime",
                        **estimate_qwen35_shared_prefix_runtime(
                            window_len=window_len,
                            concurrent_decodes=concurrency,
                            mode=mode,
                        ).to_dict(),
                    }
                )
        for concurrency in args.concurrency:
            for promoted in args.promoted:
                if promoted > concurrency:
                    continue
                rows.append(
                    {
                        "kind": "mixed_promotion_runtime",
                        **estimate_qwen35_shared_prefix_mixed_promotion(
                            window_len=window_len,
                            total_decodes=concurrency,
                            promoted_decodes=promoted,
                        ).to_dict(),
                    }
                )

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
