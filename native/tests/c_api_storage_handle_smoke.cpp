#include <cassert>
#include <cstdint>

#include "siliconrt/c_api.h"

int main() {
  siliconrt_budget_t* budget = nullptr;
  siliconrt_arena_t* arena = nullptr;

  assert(siliconrt_budget_create(8192, &budget) == SILICONRT_STATUS_OK);
  assert(siliconrt_arena_create_partitioned(6144, 2048, &arena) ==
         SILICONRT_STATUS_OK);

  siliconrt_prefix_descriptor_t prefix = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "handle",
      .logical_token_count = 128,
      .resident_token_count = 64,
      .sequence_bytes = 1024,
      .constant_bytes = 256,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &prefix, &handle) ==
         SILICONRT_STATUS_OK);

  siliconrt_storage_handle_layout_t prefix_handles = {};
  assert(siliconrt_prefix_describe_storage_handles(handle, &prefix_handles) ==
         SILICONRT_STATUS_OK);
  assert(prefix_handles.sequence.present);
  assert(prefix_handles.constant_state.present);
  assert(prefix_handles.sequence.storage_handle_id != 0);
  assert(prefix_handles.constant_state.storage_handle_id != 0);
  assert(prefix_handles.sequence.storage_handle_id !=
         prefix_handles.constant_state.storage_handle_id);

  siliconrt_decode_state_t* borrowed = nullptr;
  assert(siliconrt_decode_restore_borrowed(arena, budget, handle, &borrowed) ==
         SILICONRT_STATUS_OK);

  siliconrt_storage_handle_layout_t borrowed_handles = {};
  assert(
      siliconrt_decode_state_describe_storage_handles(borrowed, &borrowed_handles) ==
      SILICONRT_STATUS_OK);
  assert(borrowed_handles.sequence.storage_handle_id ==
         prefix_handles.sequence.storage_handle_id);
  assert(borrowed_handles.constant_state.storage_handle_id ==
         prefix_handles.constant_state.storage_handle_id);

  assert(siliconrt_decode_state_promote_sequence(borrowed) ==
         SILICONRT_STATUS_OK);

  siliconrt_storage_handle_layout_t promoted_handles = {};
  assert(
      siliconrt_decode_state_describe_storage_handles(borrowed, &promoted_handles) ==
      SILICONRT_STATUS_OK);
  assert(promoted_handles.sequence.storage_handle_id !=
         prefix_handles.sequence.storage_handle_id);
  assert(promoted_handles.constant_state.storage_handle_id ==
         prefix_handles.constant_state.storage_handle_id);

  siliconrt_decode_state_destroy(arena, budget, borrowed);
  siliconrt_prefix_destroy(arena, budget, handle);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
