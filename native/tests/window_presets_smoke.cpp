#include <cassert>

#include "siliconrt/window_presets.hpp"

int main() {
  constexpr auto safe = siliconrt::presets::qwen35_9b_text_safe();
  constexpr auto long_recall = siliconrt::presets::qwen35_9b_text_long_recall();
  constexpr auto aggressive = siliconrt::presets::qwen35_9b_text_aggressive();
  constexpr auto extreme = siliconrt::presets::qwen35_9b_text_extreme();

  assert(safe.model_key == "qwen35_9b_text");
  assert(safe.preset_name == "safe");
  assert(safe.is_bounded());
  assert(safe.policy.fixed_window_tokens().has_value());
  assert(*safe.policy.fixed_window_tokens() == 4096);
  assert(safe.tested_recall_distance_tokens == 4096);
  assert(long_recall.preset_kind == siliconrt::WindowPresetKind::kLongRecall);
  assert(*long_recall.policy.fixed_window_tokens() == 8192);
  assert(long_recall.tested_recall_distance_tokens == 8192);

  assert(aggressive.preset_kind == siliconrt::WindowPresetKind::kAggressive);
  assert(*aggressive.policy.fixed_window_tokens() == 2048);
  assert(aggressive.tested_recall_distance_tokens == 2048);

  assert(extreme.preset_kind == siliconrt::WindowPresetKind::kExtreme);
  assert(*extreme.policy.fixed_window_tokens() == 1024);
  assert(extreme.tested_recall_distance_tokens == 512);

  return 0;
}
