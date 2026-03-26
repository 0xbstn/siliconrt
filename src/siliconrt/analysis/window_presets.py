"""Model-specific bounded-window presets grounded in local probe results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class WindowPreset:
    model_key: str
    preset_name: str
    window_tokens: int
    tested_recall_distance_tokens: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def qwen35_9b_text_safe() -> WindowPreset:
    return WindowPreset(
        model_key="qwen35_9b_text",
        preset_name="safe",
        window_tokens=4096,
        tested_recall_distance_tokens=4096,
    )


def qwen35_9b_text_long_recall() -> WindowPreset:
    return WindowPreset(
        model_key="qwen35_9b_text",
        preset_name="long_recall",
        window_tokens=8192,
        tested_recall_distance_tokens=8192,
    )


def qwen35_9b_text_aggressive() -> WindowPreset:
    return WindowPreset(
        model_key="qwen35_9b_text",
        preset_name="aggressive",
        window_tokens=2048,
        tested_recall_distance_tokens=2048,
    )


def qwen35_9b_text_extreme() -> WindowPreset:
    return WindowPreset(
        model_key="qwen35_9b_text",
        preset_name="extreme",
        window_tokens=1024,
        tested_recall_distance_tokens=512,
    )
