#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_bounded_sequence.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  auto budget = siliconrt::make_budget(8192);
  auto arena = siliconrt::make_partitioned_arena(4096, 4096);

  siliconrt_prefix_descriptor_t source_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "bounded-src",
      .logical_token_count = 8,
      .resident_token_count = 8,
      .sequence_bytes = 16,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_descriptor_t destination_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "bounded-dst",
      .logical_token_count = 8,
      .resident_token_count = 8,
      .sequence_bytes = 16,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  auto source_handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), source_desc);
  auto destination_handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), destination_desc);

  siliconrt::MetalBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);
  siliconrt::MetalBufferOps ops(backend);
  siliconrt::MetalBoundedSequence bounded(&ops);

  const siliconrt::StorageSliceLayoutView source_slices{
      .layout = siliconrt::describe_prefix_storage(source_handle.get()),
      .backing_stores = stores,
  };
  const siliconrt::StorageSliceLayoutView destination_slices{
      .layout = siliconrt::describe_prefix_storage(destination_handle.get()),
      .backing_stores = stores,
  };

  auto source_sequence = backend.resolve(source_slices.sequence());
  auto destination_sequence = backend.resolve(destination_slices.sequence());

  for (std::uint8_t i = 0; i < 16; ++i) {
    source_sequence.bytes[static_cast<std::size_t>(i)] = i;
  }

  std::array<std::uint8_t, 8> append_a = {16, 17, 18, 19, 20, 21, 22, 23};
  auto result_a = bounded.append(
      source_sequence,
      12,
      destination_sequence,
      std::span<const std::uint8_t>(append_a.data(), append_a.size()));
  assert(result_a.plan.destination_used_bytes == 16);
  assert(result_a.plan.kept_source_bytes == 8);
  assert(result_a.plan.source_keep_offset_bytes == 4);
  assert(result_a.plan.destination_append_offset_bytes == 8);
  for (std::size_t i = 0; i < 8; ++i) {
    assert(destination_sequence.bytes[i] == static_cast<std::uint8_t>(4 + i));
  }
  for (std::size_t i = 0; i < 8; ++i) {
    assert(destination_sequence.bytes[8 + i] == static_cast<std::uint8_t>(16 + i));
  }

  std::array<std::uint8_t, 20> append_b = {
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  auto result_b = bounded.append(
      source_sequence,
      16,
      destination_sequence,
      std::span<const std::uint8_t>(append_b.data(), append_b.size()));
  assert(result_b.plan.kept_source_bytes == 0);
  assert(result_b.plan.append_source_offset_bytes == 4);
  assert(result_b.plan.destination_used_bytes == 16);
  for (std::size_t i = 0; i < 16; ++i) {
    assert(destination_sequence.bytes[i] == static_cast<std::uint8_t>(24 + i));
  }

  return 0;
}
