#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner bounded_planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  constexpr siliconrt::BoundedWindowPlanner unbounded_planner(
      siliconrt::profiles::qwen35_9b_text(), std::nullopt);
  siliconrt::PrefixDescriptorBuilder bounded_builder(bounded_planner);
  siliconrt::PrefixDescriptorBuilder unbounded_builder(unbounded_planner);

  const auto bounded = bounded_planner.footprint(16384);
  const auto unbounded = unbounded_planner.footprint(16384);
  const auto bounded_budget = bounded.total_bytes() * 2;

  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;
  assert(
      siliconrt_budget_create(bounded_budget, &budget) ==
      SILICONRT_STATUS_OK);
  assert(
      siliconrt_arena_create(bounded_budget, &arena) ==
      SILICONRT_STATUS_OK);

  const auto bounded_descriptor = bounded_builder.make_prefix("longprefix", 16384);
  const auto bounded_c_descriptor = bounded_descriptor.as_c_descriptor();

  siliconrt_prefix_handle_t* bounded_handle = nullptr;
  assert(
      siliconrt_prefix_create(arena, budget, &bounded_c_descriptor, &bounded_handle) ==
      SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t out = {};
  assert(siliconrt_prefix_describe(bounded_handle, &out) == SILICONRT_STATUS_OK);
  assert(out.logical_token_count == 16384);
  assert(out.resident_token_count == 2048);
  assert(out.sequence_bytes == 67108864);
  assert(out.constant_bytes == 26345472);

  siliconrt_decode_state_t* decode_state = nullptr;
  assert(
      siliconrt_decode_restore(arena, budget, bounded_handle, &decode_state) ==
      SILICONRT_STATUS_OK);
  assert(
      siliconrt_decode_state_set_residency(
          decode_state,
          16448,
          2048,
          bounded.sequence_bytes) == SILICONRT_STATUS_OK);
  siliconrt_prefix_descriptor_t decode_out = {};
  assert(
      siliconrt_decode_state_describe(decode_state, &decode_out) ==
      SILICONRT_STATUS_OK);
  assert(decode_out.logical_token_count == 16448);
  assert(decode_out.resident_token_count == 2048);
  assert(decode_out.sequence_bytes == bounded.sequence_bytes);
  assert(
      siliconrt_decode_state_set_residency(
          decode_state,
          16448,
          4096,
          bounded.sequence_bytes * 2) == SILICONRT_STATUS_OUT_OF_MEMORY);

  const auto unbounded_descriptor =
      unbounded_builder.make_prefix("longprefix-unbounded", 16384);
  const auto unbounded_c_descriptor = unbounded_descriptor.as_c_descriptor();

  siliconrt_prefix_handle_t* unbounded_handle = nullptr;
  assert(
      siliconrt_prefix_create(
          arena, budget, &unbounded_c_descriptor, &unbounded_handle) ==
      SILICONRT_STATUS_OUT_OF_MEMORY);

  siliconrt_decode_state_destroy(arena, budget, decode_state);
  siliconrt_prefix_destroy(arena, budget, bounded_handle);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
