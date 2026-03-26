#include <array>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/metal_circular_sequence.hpp"
#include "siliconrt/metal_window_descriptor.hpp"
#include "siliconrt/metal_window_gather.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/storage_slice.hpp"

int main() {
  auto budget = siliconrt::make_budget(16384);
  auto arena = siliconrt::make_partitioned_arena(8192, 4096);

  siliconrt_prefix_descriptor_t source_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "window-gather-src",
      .logical_token_count = 8,
      .resident_token_count = 8,
      .sequence_bytes = 16,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_descriptor_t destination_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "window-gather-dst",
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
  siliconrt::MetalCircularSequence circular(&ops);
  siliconrt::MetalWindowGather gather(backend);

  const siliconrt::StorageSliceLayoutView source_slices{
      .layout = siliconrt::describe_prefix_storage(source_handle.get()),
      .backing_stores = stores,
  };
  const siliconrt::StorageSliceLayoutView destination_slices{
      .layout = siliconrt::describe_prefix_storage(destination_handle.get()),
      .backing_stores = stores,
  };
  auto source = backend.resolve(source_slices.sequence());
  auto destination = backend.resolve(destination_slices.sequence());

  for (std::uint8_t i = 0; i < 16; ++i) {
    source.writable_bytes[static_cast<std::size_t>(i)] = i;
    destination.writable_bytes[static_cast<std::size_t>(i)] = 0;
  }

  siliconrt::CircularSequenceState state{
      .head_offset_bytes = 0,
      .used_bytes = 12,
      .capacity_bytes = 16,
  };
  std::array<std::uint8_t, 8> append_a = {16, 17, 18, 19, 20, 21, 22, 23};
  state = circular
              .append(
                  source,
                  state,
                  std::span<const std::uint8_t>(append_a.data(), append_a.size()))
              .plan.after;

  const auto segments = siliconrt::make_metal_window_segments(source, state);
  const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
  gather.gather(descriptor, destination);

  for (std::size_t i = 0; i < 8; ++i) {
    assert(destination.writable_bytes[i] == static_cast<std::uint8_t>(4 + i));
  }
  for (std::size_t i = 0; i < 8; ++i) {
    assert(destination.writable_bytes[8 + i] == static_cast<std::uint8_t>(16 + i));
  }

  return 0;
}
