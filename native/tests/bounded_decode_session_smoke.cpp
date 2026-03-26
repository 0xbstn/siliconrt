#include <cassert>
#include <cstdint>

#include "siliconrt/bounded_decode_session.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("session-a", 16384);
  const auto prefix_c = prefix.as_c_descriptor();

  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;
  const auto capacity = prefix.total_bytes() * 2;
  assert(siliconrt_budget_create(capacity, &budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create(capacity, &arena) == SILICONRT_STATUS_OK);

  siliconrt_prefix_handle_t* handle = nullptr;
  assert(
      siliconrt_prefix_create(arena, budget, &prefix_c, &handle) ==
      SILICONRT_STATUS_OK);

  siliconrt_decode_state_t* raw_state = nullptr;
  assert(
      siliconrt_decode_restore(arena, budget, handle, &raw_state) ==
      SILICONRT_STATUS_OK);

  siliconrt::BoundedDecodeSession session(builder, raw_state);
  auto described = session.descriptor();
  assert(described.logical_token_count == 16384);
  assert(described.resident_token_count == 2048);
  assert(described.sequence_bytes == 67108864);

  const auto delta = session.append_tokens(64);
  assert(delta.appended_tokens == 64);
  assert(delta.additional_sequence_bytes == 0);
  assert(delta.additional_total_bytes == 0);

  described = session.descriptor();
  assert(described.logical_token_count == 16448);
  assert(described.resident_token_count == 2048);
  assert(described.sequence_bytes == 67108864);

  auto* state = session.release();
  siliconrt_decode_state_destroy(arena, budget, state);
  siliconrt_prefix_destroy(arena, budget, handle);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
