"""Low-level Apple Silicon / Metal experiment package for siliconrt."""

from .analysis import (
    DTYPE_BYTES,
    KvMemoryEstimate,
    MemoryModeRequestCost,
    PagedKvEstimate,
    Qwen35TextCacheEstimate,
    block_utilization,
    bounded_kv_savings_factor,
    estimate_kv_memory,
    estimate_paged_kv,
    estimate_qwen35_9b_text_cache,
    max_sessions_under_budget,
    max_sessions_paged_under_budget,
    qwen35_request_cost,
    simulate_memory_mode_workload,
)

__all__ = [
    "__version__",
    "DTYPE_BYTES",
    "KvMemoryEstimate",
    "MemoryModeRequestCost",
    "PagedKvEstimate",
    "Qwen35TextCacheEstimate",
    "block_utilization",
    "bounded_kv_savings_factor",
    "estimate_kv_memory",
    "estimate_paged_kv",
    "estimate_qwen35_9b_text_cache",
    "max_sessions_under_budget",
    "max_sessions_paged_under_budget",
    "qwen35_request_cost",
    "simulate_memory_mode_workload",
]

try:
    from .prefix import PrefixSnapshot, PrefixStore
except ModuleNotFoundError:
    PrefixSnapshot = None
    PrefixStore = None
else:
    __all__.extend(["PrefixStore", "PrefixSnapshot"])

__version__ = "0.1.0"
