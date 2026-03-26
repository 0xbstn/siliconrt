#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_prefix_store.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto first = builder.make_prefix("partition-a", 16384);
  const auto second = builder.make_prefix("partition-b", 16384);
  const auto third = builder.make_prefix("partition-c", 16384);

  const auto total_capacity = first.total_bytes() * 2;
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, total_capacity, 2);
  assert(plan.feasible());

  siliconrt::PartitionedPrefixStore store(plan);

  auto first_handle = store.materialize(first);
  assert(first_handle.has_value());
  assert(first_handle->sequence_span_id != 0);
  assert(first_handle->constant_span_id != 0);

  auto second_handle = store.materialize(second);
  assert(second_handle.has_value());

  auto third_handle = store.materialize(third);
  assert(!third_handle.has_value());

  const auto stats = store.stats();
  assert(stats.handle_count == 2);
  assert(stats.committed_bytes == total_capacity);
  assert(stats.arena.sequence_pool.allocated_span_count == 2);
  assert(stats.arena.constant_pool.allocated_span_count == 2);

  assert(store.release(first_handle->handle_id));
  auto retry_handle = store.materialize(third);
  assert(retry_handle.has_value());
  assert(store.stats().handle_count == 2);

  return 0;
}
