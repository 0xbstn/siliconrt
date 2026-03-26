#include <cassert>
#include <cstdint>

#include "siliconrt/partitioned_runtime_plan.hpp"

int main() {
  constexpr auto profile = siliconrt::profiles::qwen35_9b_text();
  constexpr auto plan =
      siliconrt::make_sequence_biased_plan(profile, 2048, 1000000000ULL, 8);

  assert(plan.window_tokens == 2048);
  assert(plan.target_sessions == 8);
  assert(plan.per_session.sequence_bytes == 67108864ULL);
  assert(plan.per_session.constant_bytes == 26345472ULL);
  assert(plan.per_session.total_bytes == 93454336ULL);
  assert(plan.constant_capacity_bytes == 210763776ULL);
  assert(plan.sequence_capacity_bytes == 789236224ULL);
  assert(plan.max_sessions_by_constant == 8);
  assert(plan.max_sessions_by_sequence == 11);
  assert(plan.max_sessions_effective == 8);
  assert(plan.slack_bytes == 252365312ULL);
  assert(plan.feasible());

  constexpr auto infeasible =
      siliconrt::make_sequence_biased_plan(profile, 2048, 100000000ULL, 2);
  assert(!infeasible.feasible());
  assert(infeasible.max_sessions_effective == 0);

  return 0;
}
