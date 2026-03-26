#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_BIN = Path("/tmp/metal_bounded_sequence_bench")


def compile_bench(binary: Path) -> None:
    cmd = [
        "clang++",
        "-std=c++20",
        "-fobjc-arc",
        "-ObjC++",
        "-I",
        "native/include",
        "native/src/c_api.cpp",
        "native/src/metal_backing_store.mm",
        "native/src/metal_buffer_ops.mm",
        "native/src/metal_bounded_sequence.mm",
        "native/src/metal_circular_sequence.mm",
        "native/src/metal_compute_runtime.mm",
        "native/src/metal_window_linearizer.mm",
        "native/src/metal_window_gather.mm",
        "native/src/metal_window_checksum.mm",
        "native/src/metal_window_score.mm",
        "native/src/metal_window_stats.mm",
        "native/bench/metal_bounded_sequence_bench.mm",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_case(
    binary: Path,
    capacity_bytes: int,
    trim_target_bytes: int,
    append_bytes: int,
    iterations: int,
) -> dict:
    cmd = [
        str(binary),
        "--json",
        "--capacity-bytes",
        str(capacity_bytes),
        "--trim-target-bytes",
        str(trim_target_bytes),
        "--append-bytes",
        str(append_bytes),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("bench produced no JSON output")
    return json.loads(stdout_lines[-1])


def make_markdown(results: list[dict]) -> str:
    lines = [
        "# Metal Window Path Sweep",
        "",
        "| capacity bytes | append bytes | bounded append ns | circular append ns | linearize ns | gather ns | checksum ns | score direct ns | score linearized ns | stats direct ns | stats linearized ns | trim front ns |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| `{capacity_bytes}` | `{append_bytes}` | `{metal_bounded_append_ns}` | `{metal_circular_append_ns}` | `{metal_window_linearize_ns}` | `{metal_window_gather_ns}` | `{metal_window_checksum_ns}` | `{metal_window_score_direct_ns}` | `{metal_window_score_linearized_ns}` | `{metal_window_stats_direct_ns}` | `{metal_window_stats_linearized_ns}` | `{metal_trim_front_ns}` |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=DEFAULT_BENCH_BIN,
    )
    parser.add_argument(
        "--capacity-bytes",
        type=int,
        nargs="+",
        default=[131072],
    )
    parser.add_argument(
        "--trim-target-bytes",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--append-bytes",
        type=int,
        nargs="+",
        default=[512, 4096, 16384],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    compile_bench(args.bench_bin)

    results = [
        run_case(
            args.bench_bin,
            capacity_bytes,
            min(args.trim_target_bytes, capacity_bytes),
            append_bytes,
            args.iterations,
        )
        for capacity_bytes in args.capacity_bytes
        for append_bytes in args.append_bytes
    ]

    payload = {
        "capacity_bytes": args.capacity_bytes,
        "trim_target_bytes": args.trim_target_bytes,
        "iterations": args.iterations,
        "results": results,
    }

    print(json.dumps(payload, indent=2))

    if args.output_dir is not None:
        output_dir = (REPO_ROOT / args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "README.md").write_text(
            make_markdown(results),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
