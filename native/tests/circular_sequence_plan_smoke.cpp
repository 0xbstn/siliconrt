#include <array>
#include <cassert>

#include "siliconrt/circular_sequence_plan.hpp"

int main() {
  const siliconrt::CircularSequenceState base{
      .head_offset_bytes = 0,
      .used_bytes = 12,
      .capacity_bytes = 16,
  };
  const auto plan_a = siliconrt::make_circular_append_plan(base, 8);
  assert(plan_a.append_bytes == 8);
  assert(plan_a.dropped_prefix_bytes == 4);
  assert(plan_a.append_segments.first.offset_bytes == 12);
  assert(plan_a.append_segments.first.size_bytes == 4);
  assert(plan_a.append_segments.second.offset_bytes == 0);
  assert(plan_a.append_segments.second.size_bytes == 4);
  assert(plan_a.after.head_offset_bytes == 4);
  assert(plan_a.after.used_bytes == 16);

  const auto visible_a = siliconrt::make_circular_visible_segments(plan_a.after);
  assert(visible_a.first.offset_bytes == 4);
  assert(visible_a.first.size_bytes == 12);
  assert(visible_a.second.offset_bytes == 0);
  assert(visible_a.second.size_bytes == 4);

  const auto plan_b = siliconrt::make_circular_append_plan(plan_a.after, 20);
  assert(plan_b.append_source_offset_bytes == 4);
  assert(plan_b.append_bytes == 16);
  assert(plan_b.dropped_prefix_bytes == 16);
  assert(plan_b.append_segments.first.offset_bytes == 0);
  assert(plan_b.append_segments.first.size_bytes == 16);
  assert(!plan_b.append_segments.second.present());
  assert(plan_b.after.head_offset_bytes == 0);
  assert(plan_b.after.used_bytes == 16);

  return 0;
}
