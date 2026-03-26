#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_prefix_store.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cow-a", 16384);
  const auto total_capacity = prefix.total_bytes() + prefix.sequence_bytes;
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, total_capacity, 1);
  siliconrt::PartitionedPrefixStore store(plan);

  auto handle = store.materialize(prefix);
  assert(handle.has_value());
  assert(store.stats().committed_bytes == prefix.total_bytes());

  auto decode = store.restore_borrow_until_append(handle->handle_id);
  assert(decode.has_value());
  assert(decode->borrows_sequence());
  assert(!decode->owns_sequence());
  assert(decode->borrowed_sequence_span_id == handle->sequence_span_id);
  assert(store.stats().committed_bytes == prefix.total_bytes());
  assert(!store.release(handle->handle_id));

  assert(store.promote_decode_sequence(decode->handle_id));
  auto promoted = store.get_decode(decode->handle_id);
  assert(promoted.has_value());
  assert(promoted->owns_sequence());
  assert(promoted->sequence_bytes == prefix.sequence_bytes);
  assert(store.stats().committed_bytes == prefix.total_bytes() + prefix.sequence_bytes);
  assert(store.promote_decode_sequence(decode->handle_id));

  assert(store.release_decode(decode->handle_id));
  assert(store.stats().committed_bytes == prefix.total_bytes());
  assert(store.release(handle->handle_id));
  assert(store.stats().committed_bytes == 0);

  return 0;
}
