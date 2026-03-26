"""CPU-only analysis helpers for siliconrt."""

from .kv_math import (
    DTYPE_BYTES,
    KvMemoryEstimate,
    bounded_kv_savings_factor,
    estimate_kv_memory,
    max_sessions_under_budget,
)
from .model_profiles import Qwen35TextCacheEstimate, estimate_qwen35_9b_text_cache
from .paged_kv_math import (
    PagedKvEstimate,
    block_utilization,
    estimate_paged_kv,
    max_sessions_paged_under_budget,
)
from .partitioned_runtime_math import (
    PartitionedPlanEstimate,
    PartitionedSessionEstimate,
    estimate_qwen35_partitioned_session,
    make_sequence_biased_qwen35_plan,
    max_partitioned_sessions_under_budget,
)
from .restore_math import (
    RestoreAliasMode,
    RestorePlanEstimate,
    SharedPrefixConcurrencyEstimate,
    SharedPrefixMixedPromotionEstimate,
    estimate_qwen35_restore_plan,
    estimate_qwen35_shared_prefix_mixed_promotion,
    estimate_qwen35_shared_prefix_runtime,
    max_qwen35_shared_prefix_concurrency_under_budget,
)
from .workload_mix import (
    MemoryModeRequestCost,
    qwen35_request_cost,
    simulate_memory_mode_workload,
)
from .window_presets import (
    WindowPreset,
    qwen35_9b_text_aggressive,
    qwen35_9b_text_extreme,
    qwen35_9b_text_long_recall,
    qwen35_9b_text_safe,
)

__all__ = [
    "DTYPE_BYTES",
    "KvMemoryEstimate",
    "PagedKvEstimate",
    "PartitionedPlanEstimate",
    "PartitionedSessionEstimate",
    "RestoreAliasMode",
    "RestorePlanEstimate",
    "SharedPrefixConcurrencyEstimate",
    "SharedPrefixMixedPromotionEstimate",
    "WindowPreset",
    "Qwen35TextCacheEstimate",
    "MemoryModeRequestCost",
    "block_utilization",
    "bounded_kv_savings_factor",
    "estimate_kv_memory",
    "estimate_qwen35_partitioned_session",
    "estimate_qwen35_restore_plan",
    "estimate_qwen35_shared_prefix_mixed_promotion",
    "estimate_paged_kv",
    "estimate_qwen35_9b_text_cache",
    "estimate_qwen35_shared_prefix_runtime",
    "make_sequence_biased_qwen35_plan",
    "max_qwen35_shared_prefix_concurrency_under_budget",
    "max_sessions_under_budget",
    "max_partitioned_sessions_under_budget",
    "max_sessions_paged_under_budget",
    "qwen35_request_cost",
    "simulate_memory_mode_workload",
    "qwen35_9b_text_aggressive",
    "qwen35_9b_text_extreme",
    "qwen35_9b_text_long_recall",
    "qwen35_9b_text_safe",
]
