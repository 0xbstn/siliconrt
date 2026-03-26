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

  siliconrt_prefix_descriptor_t desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "trim-seq",
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
  siliconrt::MetalBoundedSequence bounded(&ops);

  const siliconrt::StorageSliceLayoutView slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  auto sequence = backend.resolve(slices.sequence());

  assert(sequence.writable_bytes.size() == 16);
  for (std::uint8_t i = 0; i < 16; ++i) {
    sequence.writable_bytes[static_cast<std::size_t>(i)] = i;
  }

  const auto trim_a = bounded.trim_front(sequence, 12, 5, 0);
  assert(trim_a.plan.trimmed_bytes == 7);
  assert(trim_a.plan.kept_bytes == 5);
  for (std::size_t i = 0; i < 5; ++i) {
    assert(sequence.writable_bytes[i] == static_cast<std::uint8_t>(7 + i));
  }
  for (std::size_t i = 5; i < 12; ++i) {
    assert(sequence.writable_bytes[i] == 0);
  }

  for (std::uint8_t i = 0; i < 16; ++i) {
    sequence.writable_bytes[static_cast<std::size_t>(i)] = i;
  }
  const auto trim_b = bounded.trim_front(sequence, 16, 0, 0xFF);
  assert(trim_b.plan.trimmed_bytes == 16);
  assert(trim_b.plan.kept_bytes == 0);
  for (std::size_t i = 0; i < 16; ++i) {
    assert(sequence.writable_bytes[i] == 0xFF);
  }

  return 0;
}
