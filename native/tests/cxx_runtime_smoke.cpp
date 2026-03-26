#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("runtime-a", 16384);
  const auto prefix_c = prefix.as_c_descriptor();

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 2);
  auto arena = siliconrt::make_arena(prefix.total_bytes() * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix_c);
  auto state =
      siliconrt::make_decode_state(arena.get(), budget.get(), handle.get());

  siliconrt_prefix_descriptor_t described = {};
  assert(
      siliconrt_decode_state_describe(state.get(), &described) ==
      SILICONRT_STATUS_OK);
  assert(described.logical_token_count == 16384);
  assert(described.resident_token_count == 2048);

  siliconrt_budget_stats_t budget_stats = {};
  assert(
      siliconrt_budget_stats(budget.get(), &budget_stats) ==
      SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == prefix.total_bytes() * 2);

  return 0;
}
