"""CPU-only paged-KV math helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class PagedKvEstimate:
    seq_len: int
    block_tokens: int
    bytes_per_token: float
    constant_bytes: float
    num_blocks: int
    block_payload_bytes: float
    payload_bytes: float
    reserved_payload_bytes: float
    wasted_payload_bytes: float
    block_table_bytes: float
    total_bytes: float
    total_gb: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def estimate_paged_kv(
    *,
    seq_len: int,
    block_tokens: int,
    bytes_per_token: float,
    constant_bytes: float = 0.0,
    block_table_bytes_per_block: int = 16,
) -> PagedKvEstimate:
    """Estimate paged-KV memory with block payload plus block-table overhead."""

    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive.")
    if bytes_per_token < 0:
        raise ValueError("bytes_per_token must be non-negative.")
    if constant_bytes < 0:
        raise ValueError("constant_bytes must be non-negative.")

    num_blocks = math.ceil(seq_len / block_tokens)
    block_payload_bytes = block_tokens * bytes_per_token
    payload_bytes = seq_len * bytes_per_token
    reserved_payload_bytes = num_blocks * block_payload_bytes
    wasted_payload_bytes = reserved_payload_bytes - payload_bytes
    block_table_bytes = num_blocks * block_table_bytes_per_block
    total_bytes = reserved_payload_bytes + block_table_bytes + constant_bytes

    return PagedKvEstimate(
        seq_len=seq_len,
        block_tokens=block_tokens,
        bytes_per_token=bytes_per_token,
        constant_bytes=constant_bytes,
        num_blocks=num_blocks,
        block_payload_bytes=block_payload_bytes,
        payload_bytes=payload_bytes,
        reserved_payload_bytes=reserved_payload_bytes,
        wasted_payload_bytes=wasted_payload_bytes,
        block_table_bytes=block_table_bytes,
        total_bytes=total_bytes,
        total_gb=total_bytes / 1e9,
    )


def max_sessions_paged_under_budget(
    *,
    budget_bytes: float,
    seq_len: int,
    block_tokens: int,
    bytes_per_token: float,
    constant_bytes: float = 0.0,
    block_table_bytes_per_block: int = 16,
) -> int:
    estimate = estimate_paged_kv(
        seq_len=seq_len,
        block_tokens=block_tokens,
        bytes_per_token=bytes_per_token,
        constant_bytes=constant_bytes,
        block_table_bytes_per_block=block_table_bytes_per_block,
    )
    if estimate.total_bytes <= 0:
        return 0
    return int(budget_bytes // estimate.total_bytes)


def block_utilization(seq_len: int, block_tokens: int) -> float:
    if seq_len <= 0 or block_tokens <= 0:
        raise ValueError("seq_len and block_tokens must be positive.")
    num_blocks = math.ceil(seq_len / block_tokens)
    return seq_len / (num_blocks * block_tokens)
