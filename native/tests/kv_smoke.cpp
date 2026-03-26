#include <cassert>
#include <cstdint>

#include "siliconrt/kv_arena.hpp"
#include "siliconrt/kv_budget.hpp"
#include "siliconrt/kv_types.hpp"

int main() {
  siliconrt::KvBudget budget(1024);
  assert(budget.can_reserve(256));
  assert(budget.commit_direct(256));
  assert(budget.committed_bytes() == 256);
  assert(budget.available_bytes() == 768);

  siliconrt::KvArena arena(4096);
  auto sequence_span =
      arena.allocate(512, 128, siliconrt::ResidencyClass::kSequenceGrowing);
  assert(sequence_span.has_value());
  assert(arena.commit(sequence_span->span_id, 256, 64));

  auto constant_span =
      arena.allocate(128, 0, siliconrt::ResidencyClass::kConstantState);
  assert(constant_span.has_value());
  assert(arena.commit(constant_span->span_id, 128, 0));

  siliconrt::PrefixHandle handle;
  handle.handle_id = 1;
  handle.model_key = "qwen35_9b";
  handle.prefix_hash_hex = "deadbeef";
  handle.cache_mode = siliconrt::CacheMode::kBoundedContiguous;
  handle.sequence_span_id = sequence_span->span_id;
  handle.constant_span_id = constant_span->span_id;
  handle.logical_token_count = 64;
  handle.resident_token_count = 64;
  handle.sequence_bytes = 256;
  handle.constant_bytes = 128;

  assert(handle.total_bytes() == 384);
  assert(!handle.has_bounded_window());
  assert(handle.compatible_with(
      "qwen35_9b", "deadbeef", siliconrt::CacheMode::kBoundedContiguous));
  assert(!handle.compatible_with(
      "other_model", "deadbeef", siliconrt::CacheMode::kBoundedContiguous));

  auto stats = arena.stats();
  assert(stats.allocated_span_count == 2);
  assert(stats.used_bytes == 384);

  assert(arena.release(sequence_span->span_id));
  assert(arena.release(constant_span->span_id));
  assert(budget.release_committed(256));

  return 0;
}
