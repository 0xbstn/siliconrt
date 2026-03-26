#pragma once

#include <cstdint>

#include "siliconrt/prefix_descriptor_builder.hpp"

namespace siliconrt {

enum class RestoreAliasMode : std::uint8_t {
  kCloneAll = 0,
  kShareConstantState = 1,
  kBorrowSequenceAndConstant = 2,
};

struct RestorePlan {
  RestoreAliasMode mode = RestoreAliasMode::kCloneAll;
  std::uint32_t logical_token_count = 0;
  std::uint32_t resident_token_count = 0;
  std::uint64_t clone_sequence_bytes = 0;
  std::uint64_t clone_constant_bytes = 0;
  std::uint64_t borrowed_sequence_bytes = 0;
  std::uint64_t borrowed_constant_bytes = 0;

  [[nodiscard]] constexpr std::uint64_t additional_bytes() const {
    return clone_sequence_bytes + clone_constant_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t visible_bytes() const {
    return additional_bytes() + borrowed_sequence_bytes + borrowed_constant_bytes;
  }

  [[nodiscard]] constexpr bool requires_sequence_promotion() const {
    return borrowed_sequence_bytes != 0 && clone_sequence_bytes == 0;
  }
};

class RestorePlanner {
 public:
  [[nodiscard]] static constexpr RestorePlan make_plan(
      const OwnedPrefixDescriptor& prefix,
      RestoreAliasMode mode) {
    RestorePlan plan;
    plan.mode = mode;
    plan.logical_token_count = prefix.logical_token_count;
    plan.resident_token_count = prefix.resident_token_count;

    switch (mode) {
      case RestoreAliasMode::kCloneAll:
        plan.clone_sequence_bytes = prefix.sequence_bytes;
        plan.clone_constant_bytes = prefix.constant_bytes;
        break;
      case RestoreAliasMode::kShareConstantState:
        plan.clone_sequence_bytes = prefix.sequence_bytes;
        plan.borrowed_constant_bytes = prefix.constant_bytes;
        break;
      case RestoreAliasMode::kBorrowSequenceAndConstant:
        plan.borrowed_sequence_bytes = prefix.sequence_bytes;
        plan.borrowed_constant_bytes = prefix.constant_bytes;
        break;
    }

    return plan;
  }
};

}  // namespace siliconrt
