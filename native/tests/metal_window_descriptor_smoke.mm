#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_descriptor.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  auto budget = siliconrt::make_budget(8192);
  auto arena = siliconrt::make_partitioned_arena(4096, 4096);

  siliconrt_prefix_descriptor_t desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "window-descriptor",
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

  const siliconrt::CircularSequenceState wrapped{
      .head_offset_bytes = 4,
      .used_bytes = 16,
      .capacity_bytes = 16,
  };
  const auto wrapped_segments = siliconrt::make_metal_window_segments(sequence, wrapped);
  const auto wrapped_desc = siliconrt::make_metal_window_descriptor(wrapped_segments);
  assert(wrapped_desc.valid());
  assert(!wrapped_desc.linear());
  assert(wrapped_desc.segment_count == 2);
  assert(wrapped_desc.first.offset_bytes == 4);
  assert(wrapped_desc.first.size_bytes == 12);
  assert(wrapped_desc.second.offset_bytes == 0);
  assert(wrapped_desc.second.size_bytes == 4);
  assert(wrapped_desc.total_bytes() == 16);

  const siliconrt::CircularSequenceState linear{
      .head_offset_bytes = 0,
      .used_bytes = 10,
      .capacity_bytes = 16,
  };
  const auto linear_segments = siliconrt::make_metal_window_segments(sequence, linear);
  const auto linear_desc = siliconrt::make_metal_window_descriptor(linear_segments);
  assert(linear_desc.valid());
  assert(linear_desc.linear());
  assert(linear_desc.segment_count == 1);
  assert(linear_desc.first.offset_bytes == 0);
  assert(linear_desc.first.size_bytes == 10);
  assert(linear_desc.second.size_bytes == 0);

  return 0;
}
