#include <algorithm>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/host_backing_store.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("host-store", 8192);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 3);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);

  siliconrt::HostBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);
  assert(backend.store_count() == 2);

  const siliconrt::StorageSliceLayoutView prefix_slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  auto prefix_sequence = backend.resolve(prefix_slices.sequence());
  auto prefix_constant = backend.resolve(prefix_slices.constant_state());
  assert(prefix_sequence.valid());
  assert(prefix_constant.valid());
  std::fill(prefix_sequence.bytes.begin(), prefix_sequence.bytes.end(), 0x11);
  std::fill(prefix_constant.bytes.begin(), prefix_constant.bytes.end(), 0x22);

  auto borrowed = siliconrt::make_borrowed_decode_state(
      arena.get(), budget.get(), handle.get());
  const siliconrt::StorageSliceLayoutView borrowed_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = stores,
  };
  auto borrowed_sequence = backend.resolve(borrowed_slices.sequence());
  auto borrowed_constant = backend.resolve(borrowed_slices.constant_state());
  assert(borrowed_sequence.valid());
  assert(borrowed_constant.valid());
  assert(!borrowed_sequence.bytes.empty());
  assert(!borrowed_constant.bytes.empty());
  assert(borrowed_sequence.bytes.data() == prefix_sequence.bytes.data());
  assert(borrowed_constant.bytes.data() == prefix_constant.bytes.data());
  assert(borrowed_sequence.bytes.front() == 0x11);
  assert(borrowed_constant.bytes.front() == 0x22);

  siliconrt::promote_decode_sequence(borrowed.get());
  const siliconrt::StorageSliceLayoutView promoted_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = stores,
  };
  auto promoted_sequence = backend.resolve(promoted_slices.sequence());
  auto promoted_constant = backend.resolve(promoted_slices.constant_state());
  assert(promoted_sequence.valid());
  assert(promoted_constant.valid());
  assert(promoted_sequence.bytes.data() != prefix_sequence.bytes.data());
  assert(promoted_constant.bytes.data() == prefix_constant.bytes.data());

  std::fill(promoted_sequence.bytes.begin(), promoted_sequence.bytes.end(), 0x33);
  assert(prefix_sequence.bytes.front() == 0x11);
  assert(promoted_sequence.bytes.front() == 0x33);
  assert(promoted_constant.bytes.front() == 0x22);

  return 0;
}
