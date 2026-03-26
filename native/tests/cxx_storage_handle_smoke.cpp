#include <cassert>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cxx-handle", 8192);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 3);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);

  const auto prefix_handles = siliconrt::describe_prefix_storage_handles(handle.get());
  assert(prefix_handles.sequence().present());
  assert(prefix_handles.constant_state().present());

  auto borrowed = siliconrt::make_borrowed_decode_state(
      arena.get(), budget.get(), handle.get());
  const auto borrowed_handles =
      siliconrt::describe_decode_storage_handles(borrowed.get());
  assert(borrowed_handles.raw.sequence.storage_handle_id ==
         prefix_handles.raw.sequence.storage_handle_id);
  assert(borrowed_handles.raw.constant_state.storage_handle_id ==
         prefix_handles.raw.constant_state.storage_handle_id);

  siliconrt::promote_decode_sequence(borrowed.get());
  const auto promoted_handles =
      siliconrt::describe_decode_storage_handles(borrowed.get());
  assert(promoted_handles.raw.sequence.storage_handle_id !=
         prefix_handles.raw.sequence.storage_handle_id);
  assert(promoted_handles.raw.constant_state.storage_handle_id ==
         prefix_handles.raw.constant_state.storage_handle_id);

  return 0;
}
