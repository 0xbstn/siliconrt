"""CPU-only restore aliasing math for bounded prefix-backed decode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json

from .partitioned_runtime_math import estimate_qwen35_partitioned_session


class RestoreAliasMode(StrEnum):
    CLONE_ALL = "clone_all"
    SHARE_CONSTANT_STATE = "share_constant_state"
    BORROW_SEQUENCE_AND_CONSTANT = "borrow_sequence_and_constant"


@dataclass(frozen=True)
class RestorePlanEstimate:
    mode: str
    window_len: int
    clone_sequence_bytes: int
    clone_constant_bytes: int
    borrowed_sequence_bytes: int
    borrowed_constant_bytes: int
    additional_bytes: int
    visible_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class SharedPrefixConcurrencyEstimate:
    mode: str
    window_len: int
    concurrent_decodes: int
    prefix_resident_bytes: int
    additional_bytes_per_decode: int
    total_runtime_bytes: int
    total_visible_bytes: int
    bytes_saved_vs_clone_all: int
    savings_fraction: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class SharedPrefixMixedPromotionEstimate:
    window_len: int
    total_decodes: int
    promoted_decodes: int
    prefix_resident_bytes: int
    promoted_sequence_bytes_per_decode: int
    total_runtime_bytes: int
    clone_all_total_runtime_bytes: int
    share_constant_total_runtime_bytes: int
    bytes_saved_vs_clone_all: int
    bytes_saved_vs_share_constant: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_qwen35_restore_plan(
    *,
    window_len: int,
    mode: str | RestoreAliasMode,
) -> RestorePlanEstimate:
    session = estimate_qwen35_partitioned_session(window_len=window_len)
    alias_mode = RestoreAliasMode(mode)

    if alias_mode is RestoreAliasMode.CLONE_ALL:
        clone_sequence_bytes = session.sequence_bytes
        clone_constant_bytes = session.constant_bytes
        borrowed_sequence_bytes = 0
        borrowed_constant_bytes = 0
    elif alias_mode is RestoreAliasMode.SHARE_CONSTANT_STATE:
        clone_sequence_bytes = session.sequence_bytes
        clone_constant_bytes = 0
        borrowed_sequence_bytes = 0
        borrowed_constant_bytes = session.constant_bytes
    elif alias_mode is RestoreAliasMode.BORROW_SEQUENCE_AND_CONSTANT:
        clone_sequence_bytes = 0
        clone_constant_bytes = 0
        borrowed_sequence_bytes = session.sequence_bytes
        borrowed_constant_bytes = session.constant_bytes
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    additional_bytes = clone_sequence_bytes + clone_constant_bytes
    visible_bytes = (
        additional_bytes + borrowed_sequence_bytes + borrowed_constant_bytes
    )
    return RestorePlanEstimate(
        mode=str(alias_mode),
        window_len=window_len,
        clone_sequence_bytes=clone_sequence_bytes,
        clone_constant_bytes=clone_constant_bytes,
        borrowed_sequence_bytes=borrowed_sequence_bytes,
        borrowed_constant_bytes=borrowed_constant_bytes,
        additional_bytes=additional_bytes,
        visible_bytes=visible_bytes,
    )


def estimate_qwen35_shared_prefix_runtime(
    *,
    window_len: int,
    concurrent_decodes: int,
    mode: str | RestoreAliasMode,
) -> SharedPrefixConcurrencyEstimate:
    if concurrent_decodes < 0:
        raise ValueError("concurrent_decodes must be non-negative.")

    session = estimate_qwen35_partitioned_session(window_len=window_len)
    plan = estimate_qwen35_restore_plan(window_len=window_len, mode=mode)
    clone_all = estimate_qwen35_restore_plan(
        window_len=window_len, mode=RestoreAliasMode.CLONE_ALL
    )

    total_runtime_bytes = (
        session.total_bytes + concurrent_decodes * plan.additional_bytes
    )
    clone_all_total_runtime_bytes = (
        session.total_bytes + concurrent_decodes * clone_all.additional_bytes
    )
    total_visible_bytes = (
        session.total_bytes + concurrent_decodes * plan.visible_bytes
    )
    bytes_saved_vs_clone_all = clone_all_total_runtime_bytes - total_runtime_bytes
    savings_fraction = (
        bytes_saved_vs_clone_all / clone_all_total_runtime_bytes
        if clone_all_total_runtime_bytes
        else 0.0
    )
    return SharedPrefixConcurrencyEstimate(
        mode=plan.mode,
        window_len=window_len,
        concurrent_decodes=concurrent_decodes,
        prefix_resident_bytes=session.total_bytes,
        additional_bytes_per_decode=plan.additional_bytes,
        total_runtime_bytes=total_runtime_bytes,
        total_visible_bytes=total_visible_bytes,
        bytes_saved_vs_clone_all=bytes_saved_vs_clone_all,
        savings_fraction=savings_fraction,
    )


def max_qwen35_shared_prefix_concurrency_under_budget(
    *,
    budget_bytes: int,
    window_len: int,
    mode: str | RestoreAliasMode,
 ) -> int | None:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive.")

    session = estimate_qwen35_partitioned_session(window_len=window_len)
    plan = estimate_qwen35_restore_plan(window_len=window_len, mode=mode)
    if budget_bytes < session.total_bytes:
        return -1
    if plan.additional_bytes == 0:
        return None
    return (budget_bytes - session.total_bytes) // plan.additional_bytes


def estimate_qwen35_shared_prefix_mixed_promotion(
    *,
    window_len: int,
    total_decodes: int,
    promoted_decodes: int,
) -> SharedPrefixMixedPromotionEstimate:
    if total_decodes < 0:
        raise ValueError("total_decodes must be non-negative.")
    if promoted_decodes < 0 or promoted_decodes > total_decodes:
        raise ValueError("promoted_decodes must be in [0, total_decodes].")

    session = estimate_qwen35_partitioned_session(window_len=window_len)
    clone_all = estimate_qwen35_restore_plan(
        window_len=window_len, mode=RestoreAliasMode.CLONE_ALL
    )
    share_constant = estimate_qwen35_restore_plan(
        window_len=window_len, mode=RestoreAliasMode.SHARE_CONSTANT_STATE
    )

    total_runtime_bytes = session.total_bytes + promoted_decodes * session.sequence_bytes
    clone_all_total_runtime_bytes = (
        session.total_bytes + total_decodes * clone_all.additional_bytes
    )
    share_constant_total_runtime_bytes = (
        session.total_bytes + total_decodes * share_constant.additional_bytes
    )
    return SharedPrefixMixedPromotionEstimate(
        window_len=window_len,
        total_decodes=total_decodes,
        promoted_decodes=promoted_decodes,
        prefix_resident_bytes=session.total_bytes,
        promoted_sequence_bytes_per_decode=session.sequence_bytes,
        total_runtime_bytes=total_runtime_bytes,
        clone_all_total_runtime_bytes=clone_all_total_runtime_bytes,
        share_constant_total_runtime_bytes=share_constant_total_runtime_bytes,
        bytes_saved_vs_clone_all=clone_all_total_runtime_bytes - total_runtime_bytes,
        bytes_saved_vs_share_constant=share_constant_total_runtime_bytes
        - total_runtime_bytes,
    )
