#include <cassert>
#include <cstdint>

#include "siliconrt/copy_on_grow_decode_session.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cow-session", 16384);
  const auto total_capacity = prefix.total_bytes() + prefix.sequence_bytes;
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, total_capacity, 1);
  siliconrt::PartitionedPrefixStore store(plan);

  auto handle = store.materialize(prefix);
  assert(handle.has_value());
  auto decode = store.restore_borrow_until_append(handle->handle_id);
  assert(decode.has_value());

  siliconrt::CopyOnGrowDecodeSession session(&store, decode->handle_id);
  assert(!session.owns_sequence());
  const auto before = session.describe();
  assert(before.borrows_sequence());
  assert(before.borrowed_sequence_bytes == prefix.sequence_bytes);

  assert(session.promote_sequence());
  assert(session.owns_sequence());
  const auto after = session.describe();
  assert(after.sequence_bytes == prefix.sequence_bytes);
  assert(after.sequence_span_id != 0);

  assert(session.release());
  assert(store.release(handle->handle_id));

  return 0;
}
