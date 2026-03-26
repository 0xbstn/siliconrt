#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"

int main() {
  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;

  assert(siliconrt_budget_create(4096, &budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create(4096, &arena) == SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t prefix = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "borrowed",
      .logical_token_count = 64,
      .resident_token_count = 32,
      .sequence_bytes = 512,
      .constant_bytes = 128,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &prefix, &handle) ==
         SILICONRT_STATUS_OK);

  siliconrt_budget_stats_t budget_stats = {};
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 640);
  assert(budget_stats.sequence_committed_bytes == 512);
  assert(budget_stats.constant_committed_bytes == 128);

  siliconrt_decode_state_t* borrowed = nullptr;
  assert(siliconrt_decode_restore_borrowed(arena, budget, handle, &borrowed) ==
         SILICONRT_STATUS_OK);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 640);
  assert(budget_stats.sequence_committed_bytes == 512);
  assert(budget_stats.constant_committed_bytes == 128);

  siliconrt_decode_state_bindings_t bindings = {};
  assert(
      siliconrt_decode_state_describe_bindings(borrowed, &bindings) ==
      SILICONRT_STATUS_OK);
  assert(!bindings.owns_sequence);
  assert(bindings.borrows_sequence);
  assert(!bindings.owns_constant);
  assert(bindings.borrows_constant);
  assert(bindings.requires_sequence_promotion);

  assert(
      siliconrt_decode_state_set_residency(
          borrowed,
          96,
          32,
          512) == SILICONRT_STATUS_INVALID_ARGUMENT);
  assert(
      siliconrt_decode_state_promote_sequence(borrowed) ==
      SILICONRT_STATUS_OK);
  assert(
      siliconrt_decode_state_describe_bindings(borrowed, &bindings) ==
      SILICONRT_STATUS_OK);
  assert(bindings.owns_sequence);
  assert(!bindings.borrows_sequence);
  assert(!bindings.owns_constant);
  assert(bindings.borrows_constant);
  assert(!bindings.requires_sequence_promotion);

  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 1152);
  assert(budget_stats.sequence_committed_bytes == 1024);
  assert(budget_stats.constant_committed_bytes == 128);
  assert(
      siliconrt_decode_state_set_residency(
          borrowed,
          96,
          32,
          512) == SILICONRT_STATUS_OK);

  siliconrt_prefix_destroy(arena, budget, handle);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 1152);

  siliconrt_decode_state_destroy(arena, budget, borrowed);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 0);

  siliconrt_arena_stats_t arena_stats = {};
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.allocated_span_count == 0);

  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
