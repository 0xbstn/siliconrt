#include <array>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  auto budget = siliconrt::make_budget(8192);
  auto arena = siliconrt::make_partitioned_arena(4096, 4096);

  siliconrt_prefix_descriptor_t desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "window-segments",
      .logical_token_count = 8,
      .resident_token_count = 8,
      .sequence_bytes = 16,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  auto handle = siliconrt::make_prefix_handle(arena.get(), budget.get(), desc);
  siliconrt::MetalBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);

  const siliconrt::StorageSliceLayoutView slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  auto sequence = backend.resolve(slices.sequence());
  for (std::uint8_t i = 0; i < 16; ++i) {
    sequence.writable_bytes[static_cast<std::size_t>(i)] = i;
  }

  const siliconrt::CircularSequenceState state{
      .head_offset_bytes = 4,
      .used_bytes = 16,
      .capacity_bytes = 16,
  };
  const auto segments = siliconrt::make_metal_window_segments(sequence, state);
  assert(segments.segment_count() == 2);
  assert(segments.total_bytes() == 16);
  assert(segments.first.offset_bytes == 4);
  assert(segments.first.size_bytes() == 12);
  assert(segments.second.offset_bytes == 0);
  assert(segments.second.size_bytes() == 4);
  for (std::size_t i = 0; i < 12; ++i) {
    assert(segments.first.bytes[i] == static_cast<std::uint8_t>(4 + i));
  }
  for (std::size_t i = 0; i < 4; ++i) {
    assert(segments.second.bytes[i] == static_cast<std::uint8_t>(i));
  }

  return 0;
}
