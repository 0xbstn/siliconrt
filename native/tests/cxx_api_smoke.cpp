#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"

int main() {
  auto budget = siliconrt::make_budget(4096);
  auto arena = siliconrt::make_arena(4096);

  siliconrt_budget_stats_t budget_stats = {};
  assert(
      siliconrt_budget_stats(budget.get(), &budget_stats) ==
      SILICONRT_STATUS_OK);
  assert(budget_stats.capacity_bytes == 4096);
  assert(budget_stats.committed_bytes == 0);

  siliconrt_arena_stats_t arena_stats = {};
  assert(
      siliconrt_arena_stats(arena.get(), &arena_stats) ==
      SILICONRT_STATUS_OK);
  assert(arena_stats.capacity_bytes == 4096);
  assert(arena_stats.allocated_span_count == 0);

  return 0;
}
