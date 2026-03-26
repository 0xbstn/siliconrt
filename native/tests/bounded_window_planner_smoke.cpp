#include <cassert>
#include <cstdint>

#include "siliconrt/bounded_window_planner.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner bounded(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  constexpr siliconrt::BoundedWindowPlanner unbounded(
      siliconrt::profiles::qwen35_9b_text(), std::nullopt);

  const auto small = bounded.footprint(1024);
  assert(small.logical_tokens == 1024);
  assert(small.resident_tokens == 1024);
  assert(!bounded.is_saturated(small));

  const auto grown = bounded.advance(small, 512);
  assert(grown.logical_tokens == 1536);
  assert(grown.resident_tokens == 1536);
  const auto grown_delta = bounded.delta_after_append(small, 512);
  assert(grown_delta.additional_sequence_bytes == 512ULL * 32768ULL);
  assert(grown_delta.additional_total_bytes == 512ULL * 32768ULL);

  const auto near_cap = bounded.footprint(1536);
  const auto saturated = bounded.advance(near_cap, 1024);
  assert(saturated.logical_tokens == 2560);
  assert(saturated.resident_tokens == 2048);
  assert(bounded.is_saturated(saturated));
  const auto sat_delta = bounded.delta_after_append(near_cap, 1024);
  assert(sat_delta.additional_sequence_bytes == 512ULL * 32768ULL);
  assert(sat_delta.additional_total_bytes == 512ULL * 32768ULL);

  const auto already_saturated = bounded.footprint(8192);
  assert(already_saturated.resident_tokens == 2048);
  const auto same_cap = bounded.advance(already_saturated, 64);
  assert(same_cap.logical_tokens == 8256);
  assert(same_cap.resident_tokens == 2048);
  const auto same_cap_delta = bounded.delta_after_append(already_saturated, 64);
  assert(same_cap_delta.additional_sequence_bytes == 0);
  assert(same_cap_delta.additional_total_bytes == 0);

  const auto unbounded_prefix = unbounded.footprint(8192);
  const auto unbounded_next = unbounded.advance(unbounded_prefix, 64);
  assert(unbounded_next.logical_tokens == 8256);
  assert(unbounded_next.resident_tokens == 8256);
  const auto unbounded_delta = unbounded.delta_after_append(unbounded_prefix, 64);
  assert(unbounded_delta.additional_sequence_bytes == 64ULL * 32768ULL);
  assert(unbounded_delta.additional_total_bytes == 64ULL * 32768ULL);

  return 0;
}
