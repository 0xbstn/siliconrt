#include <array>
#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/metal_circular_sequence.hpp"
#include "siliconrt/metal_window_descriptor.hpp"
#include "siliconrt/metal_window_score.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/storage_slice.hpp"

namespace {

std::uint64_t expected_score(const std::array<std::uint8_t, 16>& values) {
  std::uint64_t out = 0;
  for (std::uint64_t i = 0; i < values.size(); ++i) {
    const std::uint64_t value = values[static_cast<std::size_t>(i)];
    const std::uint64_t w0 = (i & 15ull) + 1ull;
    const std::uint64_t w1 = ((i >> 4) & 7ull) + 3ull;
    out += value * w0;
    out += value * value * w1;
  }
  return out;
}

}  // namespace

int main() {
  auto budget = siliconrt::make_budget(16384);
  auto arena = siliconrt::make_partitioned_arena(8192, 4096);

  siliconrt_prefix_descriptor_t source_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "window-score-src",
      .logical_token_count = 8,
      .resident_token_count = 8,
      .sequence_bytes = 16,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  auto source_handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), source_desc);

  siliconrt::MetalBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);
  siliconrt::MetalBufferOps ops(backend);
  siliconrt::MetalCircularSequence circular(&ops);
  siliconrt::MetalWindowScore score(backend);

  const siliconrt::StorageSliceLayoutView source_slices{
      .layout = siliconrt::describe_prefix_storage(source_handle.get()),
      .backing_stores = stores,
  };
  auto source = backend.resolve(source_slices.sequence());

  for (std::uint8_t i = 0; i < 16; ++i) {
    source.writable_bytes[static_cast<std::size_t>(i)] = i;
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

  const auto descriptor =
      siliconrt::make_metal_window_descriptor(
          siliconrt::make_metal_window_segments(source, state));
  const auto score_value = score.score(descriptor);

  std::array<std::uint8_t, 16> expected_values = {
      4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23};
  assert(score_value == expected_score(expected_values));
  return 0;
}
