"""Canonical KV-first benchmark matrix for dense-model work.

This does not execute any benchmark. It only emits the manifest that defines
the first benchmark program for siliconrt.
"""

from __future__ import annotations

import argparse

from .schema import BackendTarget, BenchmarkMatrix, Metric, ModelSlot, WorkloadCase


def build_matrix() -> BenchmarkMatrix:
    return BenchmarkMatrix(
        name="kv_dense_v1",
        phase="phase_1_kv_prefix_paged_bounded",
        goal=(
            "Measure whether Apple-native prefix reuse plus bounded or paged KV "
            "can beat the stock path on warm TTFT, shared-prefix concurrency, "
            "memory-bounded stability, and long-context behavior."
        ),
        backends=[
            BackendTarget(
                name="mlx_lm",
                required=True,
                notes="Primary internal baseline and default comparison target.",
            ),
            BackendTarget(
                name="siliconrt_candidate",
                required=True,
                notes="Candidate runtime or cache policy under test.",
            ),
            BackendTarget(
                name="llama_cpp_metal",
                required=False,
                notes="External Apple baseline when the same model family is available.",
            ),
            BackendTarget(
                name="vllm_metal",
                required=False,
                notes=(
                    "Compare only when the same workload is reproducible cleanly; "
                    "do not force apples-to-oranges comparisons."
                ),
            ),
        ],
        model_slots=[
            ModelSlot(
                name="dense_debug_small",
                family="dense",
                parameter_band="1B-4B",
                quantization="4bit",
                context_focus="long-context if available",
                notes="Fast iteration model for harness bring-up and bug fixing.",
            ),
            ModelSlot(
                name="dense_main",
                family="dense",
                parameter_band="7B-8B",
                quantization="4bit",
                context_focus="long-context preferred",
                notes="Primary truth-terrain model for KV-first work.",
            ),
            ModelSlot(
                name="dense_large_optional",
                family="dense",
                parameter_band="14B+",
                quantization="4bit",
                context_focus="long-context preferred",
                notes="Only after the harness is stable and the machine budget is clear.",
            ),
        ],
        workloads=[
            WorkloadCase(
                name="cold_vs_warm_prefix",
                purpose="Measure warm-prefix TTFT reduction on a single request.",
                prefix_lengths=[256, 2048, 8192, 16384],
                suffix_lengths=[16, 64, 256],
                generation_lengths=[64, 256],
                concurrencies=[1],
                cache_hit_rates=[0, 100],
                notes="Same large prefix, varying suffixes, repeated several times.",
            ),
            WorkloadCase(
                name="shared_prefix_burst",
                purpose="Measure latency and throughput under shared-prefix concurrency.",
                prefix_lengths=[2048, 8192, 16384],
                suffix_lengths=[16, 64, 256],
                generation_lengths=[64, 256],
                concurrencies=[4, 8, 16],
                cache_hit_rates=[100],
                notes="Burst several requests with the same prefix and distinct suffixes.",
            ),
            WorkloadCase(
                name="bounded_memory_pressure",
                purpose="Measure eviction and stability under a fixed memory budget.",
                prefix_lengths=[2048, 8192],
                suffix_lengths=[64],
                generation_lengths=[64, 256],
                concurrencies=[4, 8, 16],
                cache_hit_rates=[0, 50, 90],
                notes="Many distinct prefixes; force reuse, eviction, and promotion behavior.",
            ),
            WorkloadCase(
                name="long_multi_turn",
                purpose="Measure later-turn behavior as history grows.",
                prefix_lengths=[2048, 8192],
                suffix_lengths=[64, 256],
                generation_lengths=[64, 256],
                concurrencies=[1, 4],
                cache_hit_rates=[50, 90],
                notes="Growing history across 8-20 turns per session.",
            ),
            WorkloadCase(
                name="mixed_hits_misses",
                purpose="Measure cache usefulness without hiding cold-start penalties.",
                prefix_lengths=[2048, 8192],
                suffix_lengths=[16, 64, 256],
                generation_lengths=[64, 256],
                concurrencies=[4, 8, 16],
                cache_hit_rates=[0, 50, 90, 100],
                notes="Use a realistic hit-rate sweep instead of only all-hit workloads.",
            ),
        ],
        metrics=[
            Metric(name="ttft_ms", required=True, notes="Time to first token."),
            Metric(name="decode_tokens_per_sec", required=True, notes="Steady-state decode speed."),
            Metric(name="throughput_tokens_per_sec", required=True, notes="Aggregate throughput."),
            Metric(name="peak_memory_bytes", required=True, notes="Peak memory during the run."),
            Metric(name="ttft_p50_ms", required=True, notes="Median TTFT."),
            Metric(name="ttft_p95_ms", required=True, notes="Tail TTFT."),
            Metric(name="ttft_p99_ms", required=False, notes="Very-tail TTFT when sample size allows."),
            Metric(
                name="max_concurrent_sessions_before_failure",
                required=False,
                notes="Capacity under the tested memory and cache policy.",
            ),
            Metric(name="cache_hit_rate", required=False, notes="Observed hit rate, not only target hit rate."),
            Metric(name="eviction_count", required=False, notes="How often cache entries were evicted."),
        ],
        comparison_rules=[
            "Use the same model family, quantization, hardware, and prompt matrix whenever possible.",
            "Treat siliconrt vs mlx_lm as the primary A/B gate.",
            "Treat llama.cpp Metal and vllm-metal as external references, not as excuses for apples-to-oranges comparisons.",
            "Mark any non-equivalent comparison explicitly in the result note.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the canonical KV-first benchmark matrix."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON manifest.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the manifest to stdout.",
    )
    args = parser.parse_args()

    matrix = build_matrix()

    if args.out:
        matrix.write_json(args.out)

    if args.stdout or not args.out:
        print(matrix.to_json())


if __name__ == "__main__":
    main()
