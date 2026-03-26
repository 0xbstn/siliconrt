#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>

namespace siliconrt {

struct CacheFootprint {
  std::uint32_t logical_tokens = 0;
  std::uint32_t resident_tokens = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;

  [[nodiscard]] constexpr std::uint64_t total_bytes() const {
    return sequence_bytes + constant_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t saved_bytes_vs(
      const CacheFootprint& baseline) const {
    if (baseline.total_bytes() <= total_bytes()) {
      return 0;
    }
    return baseline.total_bytes() - total_bytes();
  }
};

struct CacheProfile {
  std::string_view model_key;
  std::uint32_t sequence_layer_count = 0;
  std::uint32_t constant_layer_count = 0;
  std::uint64_t sequence_bytes_per_token = 0;
  std::uint64_t constant_bytes = 0;

  [[nodiscard]] constexpr CacheFootprint footprint(
      std::uint32_t logical_tokens,
      std::optional<std::uint32_t> window_tokens = std::nullopt) const {
    const auto resident_tokens = window_tokens.has_value()
        ? std::min(logical_tokens, *window_tokens)
        : logical_tokens;

    CacheFootprint out;
    out.logical_tokens = logical_tokens;
    out.resident_tokens = resident_tokens;
    out.sequence_bytes =
        sequence_bytes_per_token * static_cast<std::uint64_t>(resident_tokens);
    out.constant_bytes = constant_bytes;
    return out;
  }
};

namespace profiles {

inline constexpr CacheProfile qwen35_9b_text() {
  return CacheProfile{
      .model_key = "qwen35_9b_text",
      .sequence_layer_count = 8,
      .constant_layer_count = 24,
      .sequence_bytes_per_token = 32768,
      .constant_bytes = 26345472,
  };
}

}  // namespace profiles

}  // namespace siliconrt
