"""CPU-only workload-mix simulations for memory mode comparisons."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json

from .model_profiles import estimate_qwen35_9b_text_cache
from .paged_kv_math import estimate_paged_kv


@dataclass(frozen=True)
class MemoryModeRequestCost:
    seq_len: int
    mode: str
    sequence_bytes: int
    constant_bytes: int
    total_bytes: int


def qwen35_request_cost(
    *,
    seq_len: int,
    mode: str,
    window_len: int | None = None,
    block_tokens: int | None = None,
    block_table_bytes_per_block: int = 16,
) -> MemoryModeRequestCost:
    if mode == "contiguous_unbounded":
        estimate = estimate_qwen35_9b_text_cache(seq_len=seq_len)
        return MemoryModeRequestCost(
            seq_len=seq_len,
            mode=mode,
            sequence_bytes=int(estimate.full_attention_total_bytes),
            constant_bytes=int(estimate.linear_attention_total_bytes),
            total_bytes=int(estimate.total_bytes),
        )

    if mode == "bounded_contiguous":
        if window_len is None:
            raise ValueError("window_len is required for bounded_contiguous.")
        estimate = estimate_qwen35_9b_text_cache(
            seq_len=seq_len,
            full_attention_window_len=window_len,
        )
        return MemoryModeRequestCost(
            seq_len=seq_len,
            mode=mode,
            sequence_bytes=int(estimate.full_attention_total_bytes),
            constant_bytes=int(estimate.linear_attention_total_bytes),
            total_bytes=int(estimate.total_bytes),
        )

    if mode == "paged":
        if block_tokens is None:
            raise ValueError("block_tokens is required for paged mode.")
        profile = estimate_qwen35_9b_text_cache(seq_len=seq_len)
        paged = estimate_paged_kv(
            seq_len=seq_len,
            block_tokens=block_tokens,
            bytes_per_token=profile.full_attention_bytes_per_token_total,
            constant_bytes=profile.linear_attention_total_bytes,
            block_table_bytes_per_block=block_table_bytes_per_block,
        )
        return MemoryModeRequestCost(
            seq_len=seq_len,
            mode=mode,
            sequence_bytes=int(paged.reserved_payload_bytes + paged.block_table_bytes),
            constant_bytes=int(paged.constant_bytes),
            total_bytes=int(paged.total_bytes),
        )

    raise ValueError(f"Unknown mode: {mode}")


def simulate_memory_mode_workload(
    *,
    budget_bytes: int,
    seq_lens: list[int],
    mode: str,
    evict_oldest: bool = False,
    window_len: int | None = None,
    block_tokens: int | None = None,
    block_table_bytes_per_block: int = 16,
) -> dict:
    used_bytes = 0
    resident: deque[MemoryModeRequestCost] = deque()
    steps = []
    evictions = 0
    admitted = 0
    rejected = 0

    for step_index, seq_len in enumerate(seq_lens, start=1):
        cost = qwen35_request_cost(
            seq_len=seq_len,
            mode=mode,
            window_len=window_len,
            block_tokens=block_tokens,
            block_table_bytes_per_block=block_table_bytes_per_block,
        )
        local_evictions = 0
        while used_bytes + cost.total_bytes > budget_bytes and evict_oldest and resident:
            victim = resident.popleft()
            used_bytes -= victim.total_bytes
            evictions += 1
            local_evictions += 1

        if used_bytes + cost.total_bytes <= budget_bytes:
            resident.append(cost)
            used_bytes += cost.total_bytes
            admitted += 1
            is_admitted = True
        else:
            rejected += 1
            is_admitted = False

        steps.append(
            {
                "step": step_index,
                "requested_seq_len": seq_len,
                "mode": mode,
                "admitted": is_admitted,
                "evicted_requests": local_evictions,
                "resident_requests": len(resident),
                "used_bytes": used_bytes,
                "available_bytes": budget_bytes - used_bytes,
                "request_total_bytes": cost.total_bytes,
            }
        )

    return {
        "budget_bytes": budget_bytes,
        "mode": mode,
        "window_len": window_len,
        "block_tokens": block_tokens,
        "block_table_bytes_per_block": block_table_bytes_per_block,
        "evict_oldest": evict_oldest,
        "admitted": admitted,
        "rejected": rejected,
        "evictions": evictions,
        "resident_requests": len(resident),
        "used_bytes": used_bytes,
        "steps": steps,
    }


def simulation_to_json(simulation: dict) -> str:
    return json.dumps(simulation, indent=2)
