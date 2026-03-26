#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cxx-storage", 8192);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 3);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);
  const auto backing_stores = siliconrt::describe_arena_backing_stores(arena.get());
  assert(backing_stores.sequence().present());
  assert(backing_stores.constant_state().present());
  assert(backing_stores.raw.sequence.kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL);
  assert(backing_stores.raw.constant_state.kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL);

  const auto prefix_layout = siliconrt::describe_prefix_storage(handle.get());
  assert(prefix_layout.sequence().owned());
  assert(prefix_layout.constant_state().owned());
  assert(prefix_layout.sequence().has_backing_store());
  assert(prefix_layout.constant_state().has_backing_store());
  assert(prefix_layout.raw.sequence.backing_store_kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL);
  assert(prefix_layout.raw.constant_state.backing_store_kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL);
  assert(prefix_layout.raw.sequence.backing_store_id ==
         backing_stores.raw.sequence.backing_store_id);
  assert(prefix_layout.raw.constant_state.backing_store_id ==
         backing_stores.raw.constant_state.backing_store_id);
  assert(prefix_layout.visible_bytes() == prefix.total_bytes());
  assert(prefix_layout.owned_bytes() == prefix.total_bytes());
  assert(prefix_layout.borrowed_bytes() == 0);

  auto borrowed = siliconrt::make_borrowed_decode_state(
      arena.get(), budget.get(), handle.get());
  auto borrowed_layout = siliconrt::describe_decode_storage(borrowed.get());
  assert(borrowed_layout.sequence().borrowed());
  assert(borrowed_layout.constant_state().borrowed());
  assert(borrowed_layout.raw.sequence.backing_store_id ==
         prefix_layout.raw.sequence.backing_store_id);
  assert(borrowed_layout.raw.constant_state.backing_store_id ==
         prefix_layout.raw.constant_state.backing_store_id);
  assert(borrowed_layout.borrowed_bytes() == prefix.total_bytes());
  assert(borrowed_layout.owned_bytes() == 0);

  siliconrt::promote_decode_sequence(borrowed.get());
  auto promoted_layout = siliconrt::describe_decode_storage(borrowed.get());
  assert(promoted_layout.sequence().owned());
  assert(promoted_layout.constant_state().borrowed());
  assert(promoted_layout.raw.sequence.backing_store_id ==
         prefix_layout.raw.sequence.backing_store_id);
  assert(promoted_layout.raw.constant_state.backing_store_id ==
         prefix_layout.raw.constant_state.backing_store_id);
  assert(promoted_layout.visible_bytes() == prefix.total_bytes());
  assert(promoted_layout.owned_bytes() == prefix.sequence_bytes);
  assert(promoted_layout.borrowed_bytes() == prefix.constant_bytes);
  assert(promoted_layout.borrows_any());

  return 0;
}
