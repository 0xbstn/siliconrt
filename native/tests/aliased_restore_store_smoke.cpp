#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_prefix_store.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("aliased-a", 16384);

  const auto total_capacity =
      prefix.total_bytes() + (2 * prefix.sequence_bytes);
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, total_capacity, 1);
  siliconrt::PartitionedPrefixStore store(plan);

  auto handle = store.materialize(prefix);
  assert(handle.has_value());
  assert(store.stats().handle_count == 1);
  assert(store.stats().decode_handle_count == 0);

  auto first_decode = store.restore_share_constant(handle->handle_id);
  auto second_decode = store.restore_share_constant(handle->handle_id);
  auto third_decode = store.restore_share_constant(handle->handle_id);
  assert(first_decode.has_value());
  assert(second_decode.has_value());
  assert(!third_decode.has_value());

  assert(first_decode->sequence_bytes == prefix.sequence_bytes);
  assert(first_decode->borrowed_constant_bytes == prefix.constant_bytes);
  assert(first_decode->borrowed_constant_span_id == handle->constant_span_id);
  assert(store.active_decode_count(handle->handle_id) == 2);
  assert(store.stats().decode_handle_count == 2);
  assert(store.stats().committed_bytes ==
         prefix.total_bytes() + (2 * prefix.sequence_bytes));

  assert(!store.release(handle->handle_id));
  assert(store.release_decode(first_decode->handle_id));
  assert(store.active_decode_count(handle->handle_id) == 1);
  assert(!store.release(handle->handle_id));

  assert(store.release_decode(second_decode->handle_id));
  assert(store.active_decode_count(handle->handle_id) == 0);
  assert(store.release(handle->handle_id));
  assert(store.stats().handle_count == 0);
  assert(store.stats().decode_handle_count == 0);
  assert(store.stats().committed_bytes == 0);

  return 0;
}
