#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("slice", 8192);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 3);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);

  const siliconrt::StorageSliceLayoutView prefix_slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = siliconrt::describe_arena_backing_stores(arena.get()),
  };

  const auto prefix_sequence = prefix_slices.sequence();
  const auto prefix_constant = prefix_slices.constant_state();
  assert(prefix_sequence.valid());
  assert(prefix_constant.valid());
  assert(prefix_sequence.owned());
  assert(prefix_constant.owned());
  assert(prefix_sequence.matches_global_mapping());
  assert(prefix_constant.matches_global_mapping());
  assert(prefix_sequence.store_relative_begin() == prefix_sequence.global_begin());
  assert(prefix_constant.global_begin() ==
         prefix_constant.backing_store.global_base_offset_bytes +
             prefix_constant.store_relative_begin());

  auto borrowed = siliconrt::make_borrowed_decode_state(
      arena.get(), budget.get(), handle.get());
  const siliconrt::StorageSliceLayoutView borrowed_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = siliconrt::describe_arena_backing_stores(arena.get()),
  };

  assert(borrowed_slices.sequence().valid());
  assert(borrowed_slices.sequence().borrowed());
  assert(borrowed_slices.constant_state().borrowed());

  siliconrt::promote_decode_sequence(borrowed.get());
  const siliconrt::StorageSliceLayoutView promoted_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = siliconrt::describe_arena_backing_stores(arena.get()),
  };
  assert(promoted_slices.sequence().valid());
  assert(promoted_slices.sequence().owned());
  assert(promoted_slices.constant_state().borrowed());
  assert(promoted_slices.sequence().matches_global_mapping());
  assert(promoted_slices.constant_state().matches_global_mapping());
  assert(promoted_slices.visible_bytes() == prefix.total_bytes());

  return 0;
}
