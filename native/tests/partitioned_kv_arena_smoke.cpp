#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_kv_arena.hpp"

int main() {
  siliconrt::PartitionedKvArena arena(
      /*sequence_capacity_bytes=*/1024,
      /*constant_capacity_bytes=*/256);

  auto seq_a = arena.allocate(
      512, 64, siliconrt::ResidencyClass::kSequenceGrowing);
  assert(seq_a.has_value());
  assert(seq_a->offset_bytes == 0);
  assert(arena.commit(seq_a->span_id, 256, 32));

  auto const_a = arena.allocate(
      128, 0, siliconrt::ResidencyClass::kConstantState);
  assert(const_a.has_value());
  assert(const_a->offset_bytes == 1024);
  assert(arena.commit(const_a->span_id, 128, 0));

  auto seq_b = arena.allocate(
      600, 64, siliconrt::ResidencyClass::kSequenceGrowing);
  assert(!seq_b.has_value());

  auto const_b = arena.allocate(
      200, 0, siliconrt::ResidencyClass::kConstantState);
  assert(!const_b.has_value());

  const auto stats = arena.stats();
  assert(stats.total_capacity_bytes == 1280);
  assert(stats.total_used_bytes == 384);
  assert(stats.total_allocated_span_count == 2);
  assert(stats.sequence_pool.capacity_bytes == 1024);
  assert(stats.constant_pool.capacity_bytes == 256);
  assert(stats.sequence_pool.used_bytes == 256);
  assert(stats.constant_pool.used_bytes == 128);

  const auto described_const = arena.get(const_a->span_id);
  assert(described_const.has_value());
  assert(described_const->offset_bytes == 1024);
  assert(described_const->used_bytes == 128);

  assert(arena.release(seq_a->span_id));
  auto seq_c = arena.allocate(
      512, 64, siliconrt::ResidencyClass::kSequenceGrowing);
  assert(seq_c.has_value());
  assert(seq_c->offset_bytes == 0);

  return 0;
}
