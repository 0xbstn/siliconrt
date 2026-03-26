#include <cassert>
#include <cstdint>

#include "siliconrt/bounded_runtime.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::BoundedRuntime runtime(
      siliconrt::PrefixDescriptorBuilder(planner),
      siliconrt::profiles::qwen35_9b_text().footprint(16384, 2048).total_bytes() * 2);

  const auto descriptor = runtime.make_prefix_descriptor("runtime-prefix", 16384);
  assert(descriptor.logical_token_count == 16384);
  assert(descriptor.resident_token_count == 2048);

  auto handle = runtime.materialize_prefix(descriptor);
  auto session = runtime.restore_decode_session(handle.get());
  auto described = session.descriptor();
  assert(described.logical_token_count == 16384);
  assert(described.resident_token_count == 2048);

  const auto delta = session.append_tokens(128);
  assert(delta.additional_sequence_bytes == 0);
  described = session.descriptor();
  assert(described.logical_token_count == 16512);
  assert(described.resident_token_count == 2048);

  siliconrt_budget_stats_t budget_stats = {};
  assert(
      siliconrt_budget_stats(runtime.budget(), &budget_stats) ==
      SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == descriptor.total_bytes() * 2);

  siliconrt_decode_state_destroy(
      runtime.arena(), runtime.budget(), session.release());
  return 0;
}
