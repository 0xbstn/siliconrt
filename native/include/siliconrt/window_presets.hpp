#pragma once

#include <cstdint>
#include <string_view>

#include "siliconrt/window_policy.hpp"

namespace siliconrt {

enum class WindowPresetKind : std::uint8_t {
  kSafe = 0,
  kAggressive = 1,
  kExtreme = 2,
  kLongRecall = 3,
};

struct NamedWindowPreset {
  std::string_view model_key;
  std::string_view preset_name;
  WindowPresetKind preset_kind = WindowPresetKind::kSafe;
  WindowPolicy policy = WindowPolicy::keep_all();
  std::uint32_t tested_recall_distance_tokens = 0;

  [[nodiscard]] constexpr bool is_bounded() const {
    return policy.fixed_window_tokens().has_value();
  }
};

namespace presets {

inline constexpr NamedWindowPreset qwen35_9b_text_safe() {
  return NamedWindowPreset{
      .model_key = "qwen35_9b_text",
      .preset_name = "safe",
      .preset_kind = WindowPresetKind::kSafe,
      .policy = WindowPolicy::fixed(4096),
      .tested_recall_distance_tokens = 4096,
  };
}

inline constexpr NamedWindowPreset qwen35_9b_text_long_recall() {
  return NamedWindowPreset{
      .model_key = "qwen35_9b_text",
      .preset_name = "long_recall",
      .preset_kind = WindowPresetKind::kLongRecall,
      .policy = WindowPolicy::fixed(8192),
      .tested_recall_distance_tokens = 8192,
  };
}

inline constexpr NamedWindowPreset qwen35_9b_text_aggressive() {
  return NamedWindowPreset{
      .model_key = "qwen35_9b_text",
      .preset_name = "aggressive",
      .preset_kind = WindowPresetKind::kAggressive,
      .policy = WindowPolicy::fixed(2048),
      .tested_recall_distance_tokens = 2048,
  };
}

inline constexpr NamedWindowPreset qwen35_9b_text_extreme() {
  return NamedWindowPreset{
      .model_key = "qwen35_9b_text",
      .preset_name = "extreme",
      .preset_kind = WindowPresetKind::kExtreme,
      .policy = WindowPolicy::fixed(1024),
      .tested_recall_distance_tokens = 512,
  };
}

}  // namespace presets

}  // namespace siliconrt
