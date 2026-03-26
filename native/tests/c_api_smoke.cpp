#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"
int main() {
  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;
  siliconrt_budget_t* other_budget = nullptr;
  siliconrt_arena_t* other_arena = nullptr;

  assert(siliconrt_budget_create(4096, &budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create(4096, &arena) == SILICONRT_STATUS_OK);
  assert(siliconrt_budget_create(2048, &other_budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create(2048, &other_arena) == SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t prefix = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "deadbeef",
      .logical_token_count = 64,
      .resident_token_count = 32,
      .sequence_bytes = 512,
      .constant_bytes = 128,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &prefix, &handle) ==
         SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t paged_prefix = prefix;
  paged_prefix.cache_mode = SILICONRT_CACHE_MODE_PAGED;
  siliconrt_prefix_handle_t* unsupported = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &paged_prefix, &unsupported) ==
         SILICONRT_STATUS_UNIMPLEMENTED);

  siliconrt_budget_stats_t budget_stats = {};
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.capacity_bytes == 4096);
  assert(budget_stats.reserved_bytes == 0);
  assert(budget_stats.committed_bytes == 640);
  assert(budget_stats.sequence_committed_bytes == 512);
  assert(budget_stats.constant_committed_bytes == 128);

  siliconrt_arena_stats_t arena_stats = {};
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(!arena_stats.is_partitioned);
  assert(arena_stats.sequence_capacity_bytes == 4096);
  assert(arena_stats.constant_capacity_bytes == 0);
  assert(arena_stats.allocated_span_count == 2);
  assert(arena_stats.used_bytes == 640);

  siliconrt_prefix_descriptor_t described = {};
  assert(siliconrt_prefix_describe(handle, &described) == SILICONRT_STATUS_OK);
  assert(described.logical_token_count == 64);
  assert(described.resident_token_count == 32);
  assert(described.sequence_bytes == 512);
  assert(described.constant_bytes == 128);

  bool compatible = false;
  assert(
      siliconrt_prefix_compatible(
          handle,
          "qwen35_9b",
          "deadbeef",
          SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
          &compatible) == SILICONRT_STATUS_OK);
  assert(compatible);
  assert(
      siliconrt_prefix_compatible(
          handle,
          "other_model",
          "deadbeef",
          SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
          &compatible) == SILICONRT_STATUS_OK);
  assert(!compatible);

  siliconrt_decode_state_t* state = nullptr;
  assert(siliconrt_decode_restore(arena, budget, handle, &state) ==
         SILICONRT_STATUS_OK);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 1280);
  assert(budget_stats.sequence_committed_bytes == 1024);
  assert(budget_stats.constant_committed_bytes == 256);
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.allocated_span_count == 4);

  siliconrt_prefix_descriptor_t decode_described = {};
  assert(
      siliconrt_decode_state_describe(state, &decode_described) ==
      SILICONRT_STATUS_OK);
  assert(decode_described.logical_token_count == 64);
  assert(decode_described.resident_token_count == 32);

  assert(
      siliconrt_decode_state_set_residency(
          state,
          96,
          32,
          512) == SILICONRT_STATUS_OK);
  assert(
      siliconrt_decode_state_describe(state, &decode_described) ==
      SILICONRT_STATUS_OK);
  assert(decode_described.logical_token_count == 96);
  assert(decode_described.resident_token_count == 32);
  assert(decode_described.sequence_bytes == 512);
  assert(
      siliconrt_decode_state_set_residency(
          state,
          96,
          64,
          1024) == SILICONRT_STATUS_OUT_OF_MEMORY);

  siliconrt_decode_state_destroy(other_arena, other_budget, state);
  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 640);
  assert(budget_stats.sequence_committed_bytes == 512);
  assert(budget_stats.constant_committed_bytes == 128);
  siliconrt_budget_stats_t other_budget_stats = {};
  assert(siliconrt_budget_stats(other_budget, &other_budget_stats) ==
         SILICONRT_STATUS_OK);
  assert(other_budget_stats.committed_bytes == 0);

  siliconrt_prefill_handle_t* prefill = nullptr;
  assert(siliconrt_prefill_begin(arena, budget, "qwen35_9b", &prefill) ==
         SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t restored = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "cafebabe",
      .logical_token_count = 32,
      .resident_token_count = 16,
      .sequence_bytes = 256,
      .constant_bytes = 64,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  siliconrt_prefix_handle_t* restored_handle = nullptr;
  assert(siliconrt_prefill_finish_as_prefix(prefill, &restored, &restored_handle) ==
         SILICONRT_STATUS_OK);

  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 960);
  assert(budget_stats.sequence_committed_bytes == 768);
  assert(budget_stats.constant_committed_bytes == 192);

  siliconrt_prefix_destroy(other_arena, other_budget, restored_handle);
  siliconrt_prefill_destroy(arena, budget, prefill);
  siliconrt_prefix_destroy(other_arena, other_budget, handle);

  assert(siliconrt_budget_stats(budget, &budget_stats) == SILICONRT_STATUS_OK);
  assert(budget_stats.committed_bytes == 0);
  assert(siliconrt_budget_stats(other_budget, &other_budget_stats) ==
         SILICONRT_STATUS_OK);
  assert(other_budget_stats.committed_bytes == 0);
  assert(siliconrt_arena_stats(arena, &arena_stats) == SILICONRT_STATUS_OK);
  assert(arena_stats.allocated_span_count == 0);
  assert(arena_stats.free_bytes == 4096);

  siliconrt_arena_destroy(other_arena);
  siliconrt_budget_destroy(other_budget);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
