"""CPU-only simulation of the first native arena and budget contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json

from .model_profiles import estimate_qwen35_9b_text_cache


@dataclass
class KvBudgetSim:
    capacity_bytes: int
    reserved_bytes: int = 0
    committed_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        return self.reserved_bytes + self.committed_bytes

    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def can_reserve(self, size_bytes: int) -> bool:
        return size_bytes <= self.available_bytes

    def commit_direct(self, size_bytes: int) -> bool:
        if not self.can_reserve(size_bytes):
            return False
        self.committed_bytes += size_bytes
        return True

    def release_committed(self, size_bytes: int) -> bool:
        if size_bytes > self.committed_bytes:
            return False
        self.committed_bytes -= size_bytes
        return True


@dataclass
class KvSpanSim:
    span_id: int
    offset_bytes: int
    capacity_bytes: int
    used_bytes: int
    token_capacity: int
    token_count: int
    residency_class: str


@dataclass
class PrefixHandleSim:
    handle_id: int
    model_key: str
    prefix_hash: str
    sequence_span_id: int
    constant_span_id: int
    token_count: int
    sequence_bytes: int
    constant_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.sequence_bytes + self.constant_bytes


class KvArenaSim:
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self._next_span_id = 1
        self._spans: dict[int, KvSpanSim] = {}
        self._free_ranges: list[tuple[int, int]] = [(0, capacity_bytes)]

    def allocate(
        self,
        *,
        capacity_bytes: int,
        token_capacity: int,
        residency_class: str,
    ) -> KvSpanSim | None:
        for index, (offset, length) in enumerate(self._free_ranges):
            if length < capacity_bytes:
                continue
            span = KvSpanSim(
                span_id=self._next_span_id,
                offset_bytes=offset,
                capacity_bytes=capacity_bytes,
                used_bytes=0,
                token_capacity=token_capacity,
                token_count=0,
                residency_class=residency_class,
            )
            self._next_span_id += 1
            self._spans[span.span_id] = span

            new_offset = offset + capacity_bytes
            new_length = length - capacity_bytes
            if new_length == 0:
                self._free_ranges.pop(index)
            else:
                self._free_ranges[index] = (new_offset, new_length)
            return span
        return None

    def commit(self, span_id: int, *, used_bytes: int, token_count: int) -> bool:
        span = self._spans.get(span_id)
        if span is None:
            return False
        if used_bytes > span.capacity_bytes or token_count > span.token_capacity:
            return False
        span.used_bytes = used_bytes
        span.token_count = token_count
        return True

    def release(self, span_id: int) -> bool:
        span = self._spans.pop(span_id, None)
        if span is None:
            return False
        self._free_ranges.append((span.offset_bytes, span.capacity_bytes))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for offset, length in self._free_ranges:
            if not merged:
                merged.append((offset, length))
                continue
            prev_offset, prev_length = merged[-1]
            if prev_offset + prev_length == offset:
                merged[-1] = (prev_offset, prev_length + length)
            else:
                merged.append((offset, length))
        self._free_ranges = merged
        return True

    def stats(self) -> dict[str, int]:
        free_bytes = sum(length for _, length in self._free_ranges)
        largest = max((length for _, length in self._free_ranges), default=0)
        allocated_capacity = sum(span.capacity_bytes for span in self._spans.values())
        used_bytes = sum(span.used_bytes for span in self._spans.values())
        return {
            "capacity_bytes": self.capacity_bytes,
            "free_bytes": free_bytes,
            "largest_free_range_bytes": largest,
            "allocated_capacity_bytes": allocated_capacity,
            "used_bytes": used_bytes,
            "allocated_span_count": len(self._spans),
        }


def allocate_qwen35_prefix_handle(
    *,
    arena: KvArenaSim,
    budget: KvBudgetSim,
    handle_id: int,
    seq_len: int,
    model_key: str = "qwen35_9b",
) -> PrefixHandleSim | None:
    estimate = estimate_qwen35_9b_text_cache(seq_len=seq_len)
    seq_bytes = int(estimate.full_attention_total_bytes)
    constant_bytes = int(estimate.linear_attention_total_bytes)
    total_bytes = seq_bytes + constant_bytes
    if not budget.commit_direct(total_bytes):
        return None

    sequence_span = arena.allocate(
        capacity_bytes=seq_bytes,
        token_capacity=seq_len,
        residency_class="sequence_growing",
    )
    if sequence_span is None:
        budget.release_committed(total_bytes)
        return None

    constant_span = arena.allocate(
        capacity_bytes=constant_bytes,
        token_capacity=0,
        residency_class="constant_state",
    )
    if constant_span is None:
        arena.release(sequence_span.span_id)
        budget.release_committed(total_bytes)
        return None

    arena.commit(sequence_span.span_id, used_bytes=seq_bytes, token_count=seq_len)
    arena.commit(constant_span.span_id, used_bytes=constant_bytes, token_count=0)
    return PrefixHandleSim(
        handle_id=handle_id,
        model_key=model_key,
        prefix_hash=f"{model_key}:{seq_len}:{handle_id}",
        sequence_span_id=sequence_span.span_id,
        constant_span_id=constant_span.span_id,
        token_count=seq_len,
        sequence_bytes=seq_bytes,
        constant_bytes=constant_bytes,
    )


def release_prefix_handle(
    *,
    arena: KvArenaSim,
    budget: KvBudgetSim,
    handle: PrefixHandleSim,
) -> None:
    arena.release(handle.sequence_span_id)
    arena.release(handle.constant_span_id)
    budget.release_committed(handle.total_bytes)


def simulate_qwen35_prefix_admission(
    *,
    budget_bytes: int,
    seq_lens: list[int],
    evict_oldest: bool = False,
) -> dict:
    budget = KvBudgetSim(capacity_bytes=budget_bytes)
    arena = KvArenaSim(capacity_bytes=budget_bytes)
    resident_handles: deque[PrefixHandleSim] = deque()
    steps = []

    for index, seq_len in enumerate(seq_lens, start=1):
        evicted = 0
        handle = allocate_qwen35_prefix_handle(
            arena=arena,
            budget=budget,
            handle_id=index,
            seq_len=seq_len,
        )
        while handle is None and evict_oldest and resident_handles:
            victim = resident_handles.popleft()
            release_prefix_handle(arena=arena, budget=budget, handle=victim)
            evicted += 1
            handle = allocate_qwen35_prefix_handle(
                arena=arena,
                budget=budget,
                handle_id=index,
                seq_len=seq_len,
            )

        admitted = handle is not None
        if handle is not None:
            resident_handles.append(handle)

        steps.append(
            {
                "step": index,
                "requested_seq_len": seq_len,
                "admitted": admitted,
                "evicted_handles": evicted,
                "resident_handles": len(resident_handles),
                "budget_used_bytes": budget.used_bytes,
                "budget_available_bytes": budget.available_bytes,
                "arena": arena.stats(),
            }
        )

    return {
        "budget_bytes": budget_bytes,
        "evict_oldest": evict_oldest,
        "final_budget": asdict(budget),
        "final_arena": arena.stats(),
        "resident_handle_count": len(resident_handles),
        "steps": steps,
    }


def simulation_to_json(simulation: dict) -> str:
    return json.dumps(simulation, indent=2)
