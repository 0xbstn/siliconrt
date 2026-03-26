#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cxx-partitioned", 16384);

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 2);
  auto arena = siliconrt::make_partitioned_arena(
      prefix.sequence_bytes * 2, prefix.constant_bytes * 2);
  auto handle = siliconrt::make_prefix_handle(
      arena.get(), budget.get(), prefix.as_c_descriptor());

  siliconrt_arena_stats_t stats = {};
  assert(siliconrt_arena_stats(arena.get(), &stats) == SILICONRT_STATUS_OK);
  assert(stats.is_partitioned);
  assert(stats.sequence_used_bytes == prefix.sequence_bytes);
  assert(stats.constant_used_bytes == prefix.constant_bytes);

  return 0;
}
