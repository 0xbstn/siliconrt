#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"
int main() {
  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;

  assert(siliconrt_budget_create(1024, &budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create(1024, &arena) == SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t large = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "aa",
      .logical_token_count = 64,
      .resident_token_count = 64,
      .sequence_bytes = 384,
      .constant_bytes = 128,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* first = nullptr;
  siliconrt_prefix_handle_t* second = nullptr;
  siliconrt_prefix_handle_t* third = nullptr;

  assert(siliconrt_prefix_create(arena, budget, &large, &first) ==
         SILICONRT_STATUS_OK);
  assert(siliconrt_prefix_create(arena, budget, &large, &second) ==
         SILICONRT_STATUS_OK);
  assert(siliconrt_prefix_create(arena, budget, &large, &third) ==
         SILICONRT_STATUS_OUT_OF_MEMORY);

  siliconrt_budget_stats_t budget_stats = {};
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 1024);

  siliconrt_prefix_destroy(arena, budget, first);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 512);

  siliconrt_prefix_descriptor_t smaller = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "bb",
      .logical_token_count = 32,
      .resident_token_count = 32,
      .sequence_bytes = 256,
      .constant_bytes = 64,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_handle_t* recovered = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &smaller, &recovered) ==
         SILICONRT_STATUS_OK);

  siliconrt_arena_stats_t arena_stats = {};
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.used_bytes == 832);
  assert(arena_stats.allocated_span_count == 4);

  siliconrt_prefix_destroy(arena, budget, recovered);
  siliconrt_prefix_destroy(arena, budget, second);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
