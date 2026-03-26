#include <array>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/circular_sequence_plan.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/metal_circular_sequence.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/storage_slice.hpp"

namespace {

}  // namespace

int main() {
  auto budget = siliconrt::make_budget(8192);
  auto arena = siliconrt::make_partitioned_arena(4096, 4096);

  siliconrt_prefix_descriptor_t desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "circular-seq",
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
  siliconrt::MetalBufferOps ops(backend);
  siliconrt::MetalCircularSequence circular(&ops);

  const siliconrt::StorageSliceLayoutView slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  auto sequence = backend.resolve(slices.sequence());
  for (std::uint8_t i = 0; i < 16; ++i) {
    sequence.writable_bytes[static_cast<std::size_t>(i)] = i;
  }

  siliconrt::CircularSequenceState state{
      .head_offset_bytes = 0,
      .used_bytes = 12,
      .capacity_bytes = 16,
  };

  std::array<std::uint8_t, 8> append_a = {16, 17, 18, 19, 20, 21, 22, 23};
  const auto result_a = circular.append(
      sequence, state, std::span<const std::uint8_t>(append_a.data(), append_a.size()));
  state = result_a.plan.after;
  assert(state.head_offset_bytes == 4);
  assert(state.used_bytes == 16);
  std::array<std::uint8_t, 16> visible_a = {};
  const auto visible_segments_a = siliconrt::make_metal_window_segments(sequence, state);
  assert(visible_segments_a.segment_count() == 2);
  std::size_t cursor_a = 0;
  for (std::size_t i = 0; i < visible_segments_a.first.size_bytes(); ++i) {
    visible_a[cursor_a++] = visible_segments_a.first.bytes[i];
  }
  for (std::size_t i = 0; i < visible_segments_a.second.size_bytes(); ++i) {
    visible_a[cursor_a++] = visible_segments_a.second.bytes[i];
  }
  for (std::size_t i = 0; i < 8; ++i) {
    assert(visible_a[i] == static_cast<std::uint8_t>(4 + i));
  }
  for (std::size_t i = 0; i < 8; ++i) {
    assert(visible_a[8 + i] == static_cast<std::uint8_t>(16 + i));
  }

  std::array<std::uint8_t, 20> append_b = {
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const auto result_b = circular.append(
      sequence, state, std::span<const std::uint8_t>(append_b.data(), append_b.size()));
  state = result_b.plan.after;
  assert(state.head_offset_bytes == 0);
  assert(state.used_bytes == 16);
  std::array<std::uint8_t, 16> visible_b = {};
  const auto visible_segments_b = siliconrt::make_metal_window_segments(sequence, state);
  assert(visible_segments_b.segment_count() == 1);
  for (std::size_t i = 0; i < visible_segments_b.first.size_bytes(); ++i) {
    visible_b[i] = visible_segments_b.first.bytes[i];
  }
  for (std::size_t i = 0; i < 16; ++i) {
    assert(visible_b[i] == static_cast<std::uint8_t>(24 + i));
  }

  return 0;
}
