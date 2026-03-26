#pragma once

#include <algorithm>
#include <cstddef>

namespace siliconrt {

struct TrimFrontPlan {
  std::size_t source_used_bytes = 0;
  std::size_t target_used_bytes = 0;
  std::size_t kept_bytes = 0;
  std::size_t trimmed_bytes = 0;
  std::size_t source_keep_offset_bytes = 0;
  std::size_t destination_used_bytes = 0;

  [[nodiscard]] constexpr bool trims() const {
    return trimmed_bytes != 0;
  }
};

[[nodiscard]] constexpr TrimFrontPlan make_trim_front_plan(
    std::size_t current_used_bytes,
    std::size_t target_used_bytes) {
  const auto kept_bytes = std::min(current_used_bytes, target_used_bytes);
  const auto trimmed_bytes = current_used_bytes - kept_bytes;
  return TrimFrontPlan{
      .source_used_bytes = current_used_bytes,
      .target_used_bytes = target_used_bytes,
      .kept_bytes = kept_bytes,
      .trimmed_bytes = trimmed_bytes,
      .source_keep_offset_bytes = trimmed_bytes,
      .destination_used_bytes = kept_bytes,
  };
}

}  // namespace siliconrt
