"""CPU-only KV memory math helpers.

These helpers are intentionally simple. They exist to estimate the first-order
memory and capacity consequences of KV design choices before native code exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "q8": 1,
    "q4": 0.5,
}


@dataclass(frozen=True)
class KvMemoryEstimate:
    layers: int
    kv_heads: int
    head_dim: int
    seq_len: int
    batch_size: int
    bytes_per_element: float
    bytes_per_token: float
    total_bytes: float
    total_gb: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def estimate_kv_memory(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype: str | None = None,
    bytes_per_element: float | None = None,
) -> KvMemoryEstimate:
    """Estimate KV cache memory usage.

    Formula:
    per_token = layers * kv_heads * head_dim * 2 * bytes_per_element

    The factor of 2 is for K and V.
    """

    if bytes_per_element is None:
        if dtype is None:
            raise ValueError("Provide either dtype or bytes_per_element.")
        try:
            bytes_per_element = DTYPE_BYTES[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown dtype: {dtype}") from exc

    bytes_per_token = layers * kv_heads * head_dim * 2 * bytes_per_element
    total_bytes = bytes_per_token * seq_len * batch_size
    return KvMemoryEstimate(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        bytes_per_element=bytes_per_element,
        bytes_per_token=bytes_per_token,
        total_bytes=total_bytes,
        total_gb=total_bytes / 1e9,
    )


def max_sessions_under_budget(
    *,
    budget_bytes: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: str | None = None,
    bytes_per_element: float | None = None,
) -> int:
    """Estimate max same-shape sessions under a byte budget."""

    estimate = estimate_kv_memory(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=1,
        dtype=dtype,
        bytes_per_element=bytes_per_element,
    )
    if estimate.total_bytes <= 0:
        return 0
    return int(budget_bytes // estimate.total_bytes)


def bounded_kv_savings_factor(*, seq_len: int, window_len: int) -> float:
    """Estimate idealized memory reduction from bounding contiguous KV."""

    if seq_len <= 0 or window_len <= 0:
        raise ValueError("seq_len and window_len must be positive.")
    effective = min(seq_len, window_len)
    return seq_len / effective
