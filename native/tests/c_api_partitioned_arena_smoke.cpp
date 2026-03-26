#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"

int main() {
  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;

  assert(siliconrt_budget_create(1280, &budget) == SILICONRT_STATUS_OK);
  assert(
      siliconrt_arena_create_partitioned(1024, 256, &arena) ==
      SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t first = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "p1",
      .logical_token_count = 64,
      .resident_token_count = 64,
      .sequence_bytes = 512,
      .constant_bytes = 128,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* first_handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &first, &first_handle) ==
         SILICONRT_STATUS_OK);
  siliconrt_budget_stats_t budget_stats = {};
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.sequence_committed_bytes == 512);
  assert(budget_stats.constant_committed_bytes == 128);

  siliconrt_arena_stats_t arena_stats = {};
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.is_partitioned);
  assert(arena_stats.sequence_capacity_bytes == 1024);
  assert(arena_stats.constant_capacity_bytes == 256);
  assert(arena_stats.sequence_used_bytes == 512);
  assert(arena_stats.constant_used_bytes == 128);

  siliconrt_prefix_descriptor_t second = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "p2",
      .logical_token_count = 32,
      .resident_token_count = 32,
      .sequence_bytes = 400,
      .constant_bytes = 128,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_handle_t* second_handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &second, &second_handle) ==
         SILICONRT_STATUS_OK);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.sequence_committed_bytes == 912);
  assert(budget_stats.constant_committed_bytes == 256);

  siliconrt_prefix_descriptor_t third = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "p3",
      .logical_token_count = 16,
      .resident_token_count = 16,
      .sequence_bytes = 64,
      .constant_bytes = 16,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_handle_t* third_handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &third, &third_handle) ==
         SILICONRT_STATUS_OUT_OF_MEMORY);

  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.sequence_used_bytes == 912);
  assert(arena_stats.constant_used_bytes == 256);
  assert(arena_stats.used_bytes == 1168);

  siliconrt_prefix_destroy(arena, budget, first_handle);
  siliconrt_prefix_destroy(arena, budget, second_handle);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
