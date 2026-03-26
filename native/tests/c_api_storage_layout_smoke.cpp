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
      .prefix_hash_hex = "layout",
      .logical_token_count = 128,
      .resident_token_count = 64,
      .sequence_bytes = 1024,
      .constant_bytes = 256,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  siliconrt_prefix_handle_t* handle = nullptr;
  assert(siliconrt_prefix_create(arena, budget, &prefix, &handle) ==
         SILICONRT_STATUS_OK);

  siliconrt_backing_store_layout_t stores = {};
  assert(siliconrt_arena_describe_backing_stores(arena, &stores) ==
         SILICONRT_STATUS_OK);
  assert(stores.sequence.present);
  assert(stores.sequence.kind == SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL);
  assert(stores.sequence.capacity_bytes == 6144);
  assert(stores.sequence.global_base_offset_bytes == 0);
  assert(stores.constant_state.present);
  assert(stores.constant_state.kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL);
  assert(stores.constant_state.capacity_bytes == 2048);
  assert(stores.constant_state.global_base_offset_bytes == 6144);

  siliconrt_storage_layout_t prefix_layout = {};
  assert(siliconrt_prefix_describe_storage(handle, &prefix_layout) ==
         SILICONRT_STATUS_OK);
  assert(prefix_layout.sequence.present);
  assert(prefix_layout.sequence.storage_kind == SILICONRT_STORAGE_KIND_SEQUENCE);
  assert(prefix_layout.sequence.ownership == SILICONRT_STORAGE_OWNERSHIP_OWNED);
  assert(prefix_layout.sequence.backing_store_kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL);
  assert(prefix_layout.sequence.backing_store_id == stores.sequence.backing_store_id);
  assert(prefix_layout.sequence.backing_store_offset_bytes ==
         prefix_layout.sequence.offset_bytes);
  assert(prefix_layout.sequence.capacity_bytes == 1024);
  assert(prefix_layout.sequence.used_bytes == 1024);
  assert(prefix_layout.sequence.token_capacity == 64);
  assert(prefix_layout.sequence.token_count == 64);
  assert(prefix_layout.constant_state.present);
  assert(prefix_layout.constant_state.storage_kind ==
         SILICONRT_STORAGE_KIND_CONSTANT);
  assert(prefix_layout.constant_state.ownership ==
         SILICONRT_STORAGE_OWNERSHIP_OWNED);
  assert(prefix_layout.constant_state.backing_store_kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL);
  assert(prefix_layout.constant_state.backing_store_id ==
         stores.constant_state.backing_store_id);
  assert(prefix_layout.constant_state.backing_store_id !=
         prefix_layout.sequence.backing_store_id);
  assert(prefix_layout.constant_state.backing_store_offset_bytes == 0);
  assert(prefix_layout.constant_state.capacity_bytes == 256);
  assert(prefix_layout.constant_state.used_bytes == 256);
  assert(prefix_layout.constant_state.token_capacity == 0);
  assert(prefix_layout.constant_state.token_count == 0);

  siliconrt_decode_state_t* borrowed = nullptr;
  assert(siliconrt_decode_restore_borrowed(arena, budget, handle, &borrowed) ==
         SILICONRT_STATUS_OK);

  siliconrt_storage_layout_t borrowed_layout = {};
  assert(siliconrt_decode_state_describe_storage(borrowed, &borrowed_layout) ==
         SILICONRT_STATUS_OK);
  assert(borrowed_layout.sequence.present);
  assert(borrowed_layout.sequence.ownership ==
         SILICONRT_STORAGE_OWNERSHIP_BORROWED);
  assert(borrowed_layout.sequence.span_id == prefix_layout.sequence.span_id);
  assert(borrowed_layout.sequence.offset_bytes ==
         prefix_layout.sequence.offset_bytes);
  assert(borrowed_layout.sequence.backing_store_id ==
         prefix_layout.sequence.backing_store_id);
  assert(borrowed_layout.constant_state.present);
  assert(borrowed_layout.constant_state.ownership ==
         SILICONRT_STORAGE_OWNERSHIP_BORROWED);
  assert(borrowed_layout.constant_state.span_id ==
         prefix_layout.constant_state.span_id);
  assert(borrowed_layout.constant_state.backing_store_id ==
         prefix_layout.constant_state.backing_store_id);

  assert(siliconrt_decode_state_promote_sequence(borrowed) ==
         SILICONRT_STATUS_OK);
  siliconrt_storage_layout_t promoted_layout = {};
  assert(siliconrt_decode_state_describe_storage(borrowed, &promoted_layout) ==
         SILICONRT_STATUS_OK);
  assert(promoted_layout.sequence.ownership ==
         SILICONRT_STORAGE_OWNERSHIP_OWNED);
  assert(promoted_layout.sequence.span_id != prefix_layout.sequence.span_id);
  assert(promoted_layout.sequence.backing_store_kind ==
         SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL);
  assert(promoted_layout.sequence.backing_store_id == stores.sequence.backing_store_id);
  assert(promoted_layout.constant_state.ownership ==
         SILICONRT_STORAGE_OWNERSHIP_BORROWED);
  assert(promoted_layout.constant_state.span_id ==
         prefix_layout.constant_state.span_id);

  siliconrt_decode_state_destroy(arena, budget, borrowed);
  siliconrt_prefix_destroy(arena, budget, handle);
  siliconrt_arena_destroy(arena);
  siliconrt_budget_destroy(budget);
  return 0;
}
