"""Small schema for benchmark planning manifests.

This module is intentionally lightweight and stdlib-only. It exists so we can
check in canonical benchmark matrices before running heavyweight GPU work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendTarget:
    name: str
    required: bool
    notes: str


@dataclass(frozen=True)
class ModelSlot:
    name: str
    family: str
    parameter_band: str
    quantization: str
    context_focus: str
    notes: str


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    purpose: str
    prefix_lengths: list[int]
    suffix_lengths: list[int]
    generation_lengths: list[int]
    concurrencies: list[int]
    cache_hit_rates: list[int]
    notes: str


@dataclass(frozen=True)
class Metric:
    name: str
    required: bool
    notes: str


@dataclass(frozen=True)
class BenchmarkMatrix:
    name: str
    phase: str
    goal: str
    backends: list[BackendTarget]
    model_slots: list[ModelSlot]
    workloads: list[WorkloadCase]
    metrics: list[Metric]
    comparison_rules: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
