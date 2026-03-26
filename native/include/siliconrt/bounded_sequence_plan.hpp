#pragma once

#include <cstddef>
#include <cstdint>

namespace siliconrt {

struct BoundedAppendPlan {
  std::size_t source_keep_offset_bytes = 0;
  std::size_t kept_source_bytes = 0;
  std::size_t append_source_offset_bytes = 0;
  std::size_t append_bytes = 0;
  std::size_t destination_append_offset_bytes = 0;
  std::size_t destination_used_bytes = 0;

  [[nodiscard]] constexpr bool requires_tail_copy() const {
    return kept_source_bytes != 0;
  }
};

[[nodiscard]] constexpr BoundedAppendPlan make_bounded_append_plan(
    std::size_t current_used_bytes,
    std::size_t append_bytes,
    std::size_t capacity_bytes) {
  BoundedAppendPlan plan;
  if (capacity_bytes == 0) {
    return plan;
  }

  const auto bounded_append_bytes =
      append_bytes >= capacity_bytes ? capacity_bytes : append_bytes;
  plan.append_bytes = bounded_append_bytes;
  plan.append_source_offset_bytes =
      append_bytes > capacity_bytes ? append_bytes - capacity_bytes : 0;

  if (bounded_append_bytes >= capacity_bytes) {
    plan.destination_append_offset_bytes = 0;
    plan.destination_used_bytes = capacity_bytes;
    return plan;
  }

  const auto keep_budget = capacity_bytes - bounded_append_bytes;
  const auto kept_source_bytes =
      current_used_bytes >= keep_budget ? keep_budget : current_used_bytes;

  plan.kept_source_bytes = kept_source_bytes;
  plan.source_keep_offset_bytes =
      current_used_bytes > kept_source_bytes ? current_used_bytes - kept_source_bytes : 0;
  plan.destination_append_offset_bytes = kept_source_bytes;
  plan.destination_used_bytes = kept_source_bytes + bounded_append_bytes;
  return plan;
}

}  // namespace siliconrt
