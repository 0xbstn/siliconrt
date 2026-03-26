#include <algorithm>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("metal-ops", 8192);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 3);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);

  siliconrt::MetalBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);
  siliconrt::MetalBufferOps ops(backend);

  const siliconrt::StorageSliceLayoutView prefix_slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  auto prefix_sequence = backend.resolve(prefix_slices.sequence());
  auto prefix_constant = backend.resolve(prefix_slices.constant_state());
  ops.fill(prefix_sequence, 0x12);
  ops.fill(prefix_constant, 0x34);
  assert(prefix_sequence.bytes.front() == 0x12);
  assert(prefix_constant.bytes.front() == 0x34);

  auto borrowed = siliconrt::make_borrowed_decode_state(
      arena.get(), budget.get(), handle.get());
  const siliconrt::StorageSliceLayoutView borrowed_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = stores,
  };
  auto borrowed_sequence = backend.resolve(borrowed_slices.sequence());
  auto borrowed_constant = backend.resolve(borrowed_slices.constant_state());
  assert(borrowed_sequence.bytes.front() == 0x12);
  assert(borrowed_constant.bytes.front() == 0x34);

  siliconrt::promote_decode_sequence(borrowed.get());
  const siliconrt::StorageSliceLayoutView promoted_slices{
      .layout = siliconrt::describe_decode_storage(borrowed.get()),
      .backing_stores = stores,
  };
  auto promoted_sequence = backend.resolve(promoted_slices.sequence());
  auto promoted_constant = backend.resolve(promoted_slices.constant_state());
  assert(promoted_sequence.bytes.data() != prefix_sequence.bytes.data());
  assert(promoted_constant.bytes.data() == prefix_constant.bytes.data());

  ops.copy(prefix_sequence, promoted_sequence, prefix_sequence.bytes.size());
  assert(promoted_sequence.bytes.front() == 0x12);
  ops.fill(promoted_sequence, 0x56);
  assert(promoted_sequence.bytes.front() == 0x56);
  assert(prefix_sequence.bytes.front() == 0x12);
  assert(promoted_constant.bytes.front() == 0x34);

  for (std::uint8_t i = 0; i < 16; ++i) {
    promoted_sequence.writable_bytes[static_cast<std::size_t>(i)] = i;
  }
  ops.copy_region(promoted_sequence, 0, promoted_sequence, 4, 8);
  for (std::size_t i = 0; i < 4; ++i) {
    assert(promoted_sequence.writable_bytes[i] == static_cast<std::uint8_t>(i));
  }
  for (std::size_t i = 0; i < 8; ++i) {
    assert(promoted_sequence.writable_bytes[4 + i] == static_cast<std::uint8_t>(i));
  }

  ops.fill_region(promoted_sequence, 12, 4, 0xAA);
  for (std::size_t i = 12; i < 16; ++i) {
    assert(promoted_sequence.writable_bytes[i] == 0xAA);
  }

  return 0;
}
