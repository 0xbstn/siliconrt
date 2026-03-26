#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_prefix_store.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto sample = builder.make_prefix("runtime-sample", 16384);
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, sample.total_bytes() * 2, 2);

  siliconrt::PartitionedBoundedRuntime runtime(builder, plan);
  const auto first = runtime.make_prefix_descriptor("runtime-a", 16384);
  const auto second = runtime.make_prefix_descriptor("runtime-b", 16384);
  const auto third = runtime.make_prefix_descriptor("runtime-c", 16384);

  auto first_handle = runtime.materialize_prefix(first);
  auto second_handle = runtime.materialize_prefix(second);
  auto third_handle = runtime.materialize_prefix(third);

  assert(first_handle.has_value());
  assert(second_handle.has_value());
  assert(!third_handle.has_value());
  assert(runtime.store().stats().handle_count == 2);
  assert(runtime.release_prefix(first_handle->handle_id));
  assert(runtime.materialize_prefix(third).has_value());

  return 0;
}
