#pragma once

#include <cstdint>
#include <string>

namespace siliconrt {

enum class CacheMode : std::uint8_t {
  kUnknown = 0,
  kBoundedContiguous = 1,
  kPaged = 2,
};

enum class ResidencyClass : std::uint8_t {
  kUnknown = 0,
  kSequenceGrowing = 1,
  kConstantState = 2,
};

struct KvSpan {
  std::uint64_t span_id = 0;
  std::uint64_t offset_bytes = 0;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t used_bytes = 0;
  std::uint32_t token_capacity = 0;
  std::uint32_t token_count = 0;
  ResidencyClass residency_class = ResidencyClass::kUnknown;
  bool in_use = false;
};

struct PrefixHandle {
  std::uint64_t handle_id = 0;
  std::string model_key;
  std::string prefix_hash_hex;
  CacheMode cache_mode = CacheMode::kBoundedContiguous;
  std::uint64_t sequence_span_id = 0;
  std::uint64_t constant_span_id = 0;
  std::uint32_t logical_token_count = 0;
  std::uint32_t resident_token_count = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
  std::uint32_t abi_version = 1;

  [[nodiscard]] std::uint64_t total_bytes() const {
    return sequence_bytes + constant_bytes;
  }

  [[nodiscard]] bool has_bounded_window() const {
    return resident_token_count < logical_token_count;
  }

  [[nodiscard]] bool compatible_with(
      const std::string& candidate_model_key,
      const std::string& candidate_prefix_hash_hex,
      CacheMode candidate_mode) const {
    return abi_version == 1 &&
           model_key == candidate_model_key &&
           prefix_hash_hex == candidate_prefix_hash_hex &&
           cache_mode == candidate_mode;
  }
};

}  // namespace siliconrt
