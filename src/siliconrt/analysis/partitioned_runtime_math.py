"""CPU-only split-planning helpers for partitioned sequence/constant KV pools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .model_profiles import estimate_qwen35_9b_text_cache


@dataclass(frozen=True)
class PartitionedSessionEstimate:
    window_len: int
    sequence_bytes: int
    constant_bytes: int
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PartitionedPlanEstimate:
    budget_bytes: int
    window_len: int
    target_sessions: int
    sequence_capacity_bytes: int
    constant_capacity_bytes: int
    slack_bytes: int
    max_sessions_by_sequence: int
    max_sessions_by_constant: int
    max_sessions_effective: int
    per_session: PartitionedSessionEstimate

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        out = asdict(self)
        out["per_session"] = self.per_session.to_dict()
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def estimate_qwen35_partitioned_session(*, window_len: int) -> PartitionedSessionEstimate:
    if window_len <= 0:
        raise ValueError("window_len must be positive.")

    estimate = estimate_qwen35_9b_text_cache(
        seq_len=window_len,
        full_attention_window_len=window_len,
    )
    return PartitionedSessionEstimate(
        window_len=window_len,
        sequence_bytes=int(estimate.full_attention_total_bytes),
        constant_bytes=int(estimate.linear_attention_total_bytes),
        total_bytes=int(estimate.total_bytes),
    )


def max_partitioned_sessions_under_budget(*, budget_bytes: int, window_len: int) -> int:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive.")
    session = estimate_qwen35_partitioned_session(window_len=window_len)
    return budget_bytes // session.total_bytes


def make_sequence_biased_qwen35_plan(
    *,
    budget_bytes: int,
    window_len: int,
    target_sessions: int,
) -> PartitionedPlanEstimate:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive.")
    if target_sessions < 0:
        raise ValueError("target_sessions must be non-negative.")

    session = estimate_qwen35_partitioned_session(window_len=window_len)
    required_constant = session.constant_bytes * target_sessions
    required_sequence = session.sequence_bytes * target_sessions
    required_total = required_constant + required_sequence

    constant_capacity = min(required_constant, budget_bytes)
    sequence_capacity = budget_bytes - constant_capacity
    max_sessions_by_sequence = (
        sequence_capacity // session.sequence_bytes if session.sequence_bytes else 0
    )
    max_sessions_by_constant = (
        constant_capacity // session.constant_bytes if session.constant_bytes else 0
    )
    return PartitionedPlanEstimate(
        budget_bytes=budget_bytes,
        window_len=window_len,
        target_sessions=target_sessions,
        sequence_capacity_bytes=sequence_capacity,
        constant_capacity_bytes=constant_capacity,
        slack_bytes=max(0, budget_bytes - required_total),
        max_sessions_by_sequence=max_sessions_by_sequence,
        max_sessions_by_constant=max_sessions_by_constant,
        max_sessions_effective=min(max_sessions_by_sequence, max_sessions_by_constant),
        per_session=session,
    )
