#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "siliconrt/cache_profile.hpp"
#include "siliconrt/window_policy.hpp"

namespace siliconrt {

struct ResidencyDelta {
  std::uint32_t appended_tokens = 0;
  std::uint64_t additional_sequence_bytes = 0;
  std::uint64_t additional_total_bytes = 0;
};

class BoundedWindowPlanner {
 public:
  constexpr BoundedWindowPlanner(
      CacheProfile profile,
      std::optional<std::uint32_t> window_tokens)
      : BoundedWindowPlanner(
            profile,
            window_tokens.has_value()
                ? WindowPolicy::fixed(*window_tokens)
                : WindowPolicy::keep_all()) {}

  constexpr BoundedWindowPlanner(
      CacheProfile profile,
      WindowPolicy window_policy)
      : profile_(profile), window_policy_(window_policy) {}

  [[nodiscard]] constexpr CacheProfile profile() const { return profile_; }

  [[nodiscard]] constexpr std::optional<std::uint32_t> window_tokens() const {
    return window_policy_.fixed_window_tokens();
  }

  [[nodiscard]] constexpr WindowPolicy window_policy() const {
    return window_policy_;
  }

  [[nodiscard]] constexpr CacheFootprint footprint(
      std::uint32_t logical_tokens) const {
    return profile_.footprint(logical_tokens, window_policy_.fixed_window_tokens());
  }

  [[nodiscard]] constexpr CacheFootprint advance(
      const CacheFootprint& current,
      std::uint32_t appended_tokens) const {
    return footprint(current.logical_tokens + appended_tokens);
  }

  [[nodiscard]] constexpr ResidencyDelta delta_after_append(
      const CacheFootprint& current,
      std::uint32_t appended_tokens) const {
    const auto next = advance(current, appended_tokens);
    ResidencyDelta out;
    out.appended_tokens = appended_tokens;
    if (next.sequence_bytes > current.sequence_bytes) {
      out.additional_sequence_bytes = next.sequence_bytes - current.sequence_bytes;
    }
    if (next.total_bytes() > current.total_bytes()) {
      out.additional_total_bytes = next.total_bytes() - current.total_bytes();
    }
    return out;
  }

  [[nodiscard]] constexpr bool is_saturated(
      const CacheFootprint& footprint) const {
    return window_policy_.is_saturated(footprint.resident_tokens);
  }

 private:
  CacheProfile profile_;
  WindowPolicy window_policy_ = WindowPolicy::keep_all();
};

}  // namespace siliconrt
