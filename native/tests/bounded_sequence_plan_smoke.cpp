#include <cassert>

#include "siliconrt/bounded_sequence_plan.hpp"

int main() {
  {
    constexpr auto plan = siliconrt::make_bounded_append_plan(6, 2, 10);
    static_assert(plan.source_keep_offset_bytes == 0);
    static_assert(plan.kept_source_bytes == 6);
    static_assert(plan.append_source_offset_bytes == 0);
    static_assert(plan.append_bytes == 2);
    static_assert(plan.destination_append_offset_bytes == 6);
    static_assert(plan.destination_used_bytes == 8);
  }

  {
    constexpr auto plan = siliconrt::make_bounded_append_plan(10, 4, 10);
    static_assert(plan.source_keep_offset_bytes == 4);
    static_assert(plan.kept_source_bytes == 6);
    static_assert(plan.append_source_offset_bytes == 0);
    static_assert(plan.append_bytes == 4);
    static_assert(plan.destination_append_offset_bytes == 6);
    static_assert(plan.destination_used_bytes == 10);
  }

  {
    constexpr auto plan = siliconrt::make_bounded_append_plan(10, 14, 10);
    static_assert(plan.kept_source_bytes == 0);
    static_assert(plan.append_source_offset_bytes == 4);
    static_assert(plan.append_bytes == 10);
    static_assert(plan.destination_append_offset_bytes == 0);
    static_assert(plan.destination_used_bytes == 10);
  }

  return 0;
}
