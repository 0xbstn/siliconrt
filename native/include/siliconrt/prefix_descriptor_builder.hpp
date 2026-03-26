#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "siliconrt/bounded_window_planner.hpp"
#include "siliconrt/c_api.h"

namespace siliconrt {

struct OwnedPrefixDescriptor {
  std::string model_key;
  std::string prefix_hash_hex;
  std::uint32_t logical_token_count = 0;
  std::uint32_t resident_token_count = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
  siliconrt_cache_mode_t cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS;

  [[nodiscard]] std::uint64_t total_bytes() const {
    return sequence_bytes + constant_bytes;
  }

  [[nodiscard]] siliconrt_prefix_descriptor_t as_c_descriptor() const {
    return siliconrt_prefix_descriptor_t{
        .model_key = model_key.c_str(),
        .prefix_hash_hex = prefix_hash_hex.c_str(),
        .logical_token_count = logical_token_count,
        .resident_token_count = resident_token_count,
        .sequence_bytes = sequence_bytes,
        .constant_bytes = constant_bytes,
        .cache_mode = cache_mode,
    };
  }
};

class PrefixDescriptorBuilder {
 public:
  explicit constexpr PrefixDescriptorBuilder(BoundedWindowPlanner planner)
      : planner_(planner) {}

  [[nodiscard]] constexpr const BoundedWindowPlanner& planner() const {
    return planner_;
  }

  [[nodiscard]] OwnedPrefixDescriptor make_prefix(
      std::string prefix_hash_hex,
      std::uint32_t logical_token_count) const {
    return make_from_footprint(
        std::move(prefix_hash_hex), planner_.footprint(logical_token_count));
  }

  [[nodiscard]] OwnedPrefixDescriptor advance(
      const OwnedPrefixDescriptor& current,
      std::uint32_t appended_tokens) const {
    CacheFootprint current_footprint;
    current_footprint.logical_tokens = current.logical_token_count;
    current_footprint.resident_tokens = current.resident_token_count;
    current_footprint.sequence_bytes = current.sequence_bytes;
    current_footprint.constant_bytes = current.constant_bytes;

    return make_from_footprint(
        current.prefix_hash_hex,
        planner_.advance(current_footprint, appended_tokens));
  }

  [[nodiscard]] ResidencyDelta delta_after_append(
      const OwnedPrefixDescriptor& current,
      std::uint32_t appended_tokens) const {
    CacheFootprint current_footprint;
    current_footprint.logical_tokens = current.logical_token_count;
    current_footprint.resident_tokens = current.resident_token_count;
    current_footprint.sequence_bytes = current.sequence_bytes;
    current_footprint.constant_bytes = current.constant_bytes;
    return planner_.delta_after_append(current_footprint, appended_tokens);
  }

 private:
  [[nodiscard]] OwnedPrefixDescriptor make_from_footprint(
      std::string prefix_hash_hex,
      const CacheFootprint& footprint) const {
    return OwnedPrefixDescriptor{
        .model_key = std::string(planner_.profile().model_key),
        .prefix_hash_hex = std::move(prefix_hash_hex),
        .logical_token_count = footprint.logical_tokens,
        .resident_token_count = footprint.resident_tokens,
        .sequence_bytes = footprint.sequence_bytes,
        .constant_bytes = footprint.constant_bytes,
        .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
    };
  }

  BoundedWindowPlanner planner_;
};

}  // namespace siliconrt
