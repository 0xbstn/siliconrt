#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct siliconrt_budget siliconrt_budget_t;
typedef struct siliconrt_arena siliconrt_arena_t;
typedef struct siliconrt_prefix_handle siliconrt_prefix_handle_t;
typedef struct siliconrt_decode_state siliconrt_decode_state_t;
typedef struct siliconrt_prefill_handle siliconrt_prefill_handle_t;

typedef enum siliconrt_status {
  SILICONRT_STATUS_OK = 0,
  SILICONRT_STATUS_INVALID_ARGUMENT = 1,
  SILICONRT_STATUS_OUT_OF_MEMORY = 2,
  SILICONRT_STATUS_NOT_FOUND = 3,
  SILICONRT_STATUS_INCOMPATIBLE = 4,
  SILICONRT_STATUS_UNIMPLEMENTED = 5,
} siliconrt_status_t;

typedef enum siliconrt_cache_mode {
  SILICONRT_CACHE_MODE_UNKNOWN = 0,
  SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS = 1,
  SILICONRT_CACHE_MODE_PAGED = 2,
} siliconrt_cache_mode_t;

typedef enum siliconrt_storage_kind {
  SILICONRT_STORAGE_KIND_UNKNOWN = 0,
  SILICONRT_STORAGE_KIND_SEQUENCE = 1,
  SILICONRT_STORAGE_KIND_CONSTANT = 2,
} siliconrt_storage_kind_t;

typedef enum siliconrt_storage_ownership {
  SILICONRT_STORAGE_OWNERSHIP_UNKNOWN = 0,
  SILICONRT_STORAGE_OWNERSHIP_OWNED = 1,
  SILICONRT_STORAGE_OWNERSHIP_BORROWED = 2,
} siliconrt_storage_ownership_t;

typedef enum siliconrt_backing_store_kind {
  SILICONRT_BACKING_STORE_KIND_UNKNOWN = 0,
  SILICONRT_BACKING_STORE_KIND_HOST_UNIFIED = 1,
  SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL = 2,
  SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL = 3,
} siliconrt_backing_store_kind_t;

typedef struct siliconrt_budget_stats {
  uint64_t capacity_bytes;
  uint64_t reserved_bytes;
  uint64_t committed_bytes;
  uint64_t sequence_reserved_bytes;
  uint64_t constant_reserved_bytes;
  uint64_t sequence_committed_bytes;
  uint64_t constant_committed_bytes;
} siliconrt_budget_stats_t;

typedef struct siliconrt_arena_stats {
  uint64_t capacity_bytes;
  uint64_t free_bytes;
  uint64_t largest_free_range_bytes;
  uint64_t allocated_capacity_bytes;
  uint64_t used_bytes;
  uint64_t allocated_span_count;
  bool is_partitioned;
  uint64_t sequence_capacity_bytes;
  uint64_t constant_capacity_bytes;
  uint64_t sequence_used_bytes;
  uint64_t constant_used_bytes;
} siliconrt_arena_stats_t;

typedef struct siliconrt_prefix_descriptor {
  const char* model_key;
  const char* prefix_hash_hex;
  uint32_t logical_token_count;
  uint32_t resident_token_count;
  uint64_t sequence_bytes;
  uint64_t constant_bytes;
  siliconrt_cache_mode_t cache_mode;
} siliconrt_prefix_descriptor_t;

typedef struct siliconrt_storage_binding {
  bool present;
  siliconrt_storage_kind_t storage_kind;
  siliconrt_storage_ownership_t ownership;
  siliconrt_backing_store_kind_t backing_store_kind;
  uint64_t backing_store_id;
  uint64_t backing_store_offset_bytes;
  uint64_t span_id;
  uint64_t offset_bytes;
  uint64_t capacity_bytes;
  uint64_t used_bytes;
  uint32_t token_capacity;
  uint32_t token_count;
} siliconrt_storage_binding_t;

typedef struct siliconrt_storage_layout {
  siliconrt_storage_binding_t sequence;
  siliconrt_storage_binding_t constant_state;
} siliconrt_storage_layout_t;

typedef struct siliconrt_backing_store_descriptor {
  bool present;
  siliconrt_backing_store_kind_t kind;
  uint64_t backing_store_id;
  uint64_t capacity_bytes;
  uint64_t global_base_offset_bytes;
} siliconrt_backing_store_descriptor_t;

typedef struct siliconrt_backing_store_layout {
  siliconrt_backing_store_descriptor_t sequence;
  siliconrt_backing_store_descriptor_t constant_state;
} siliconrt_backing_store_layout_t;

typedef struct siliconrt_storage_handle_descriptor {
  bool present;
  siliconrt_storage_kind_t storage_kind;
  uint64_t storage_handle_id;
  siliconrt_backing_store_kind_t backing_store_kind;
  uint64_t backing_store_id;
  uint64_t backing_store_offset_bytes;
  uint64_t capacity_bytes;
  uint32_t token_capacity;
} siliconrt_storage_handle_descriptor_t;

typedef struct siliconrt_storage_handle_layout {
  siliconrt_storage_handle_descriptor_t sequence;
  siliconrt_storage_handle_descriptor_t constant_state;
} siliconrt_storage_handle_layout_t;

typedef struct siliconrt_decode_state_bindings {
  bool owns_sequence;
  bool borrows_sequence;
  bool owns_constant;
  bool borrows_constant;
  bool requires_sequence_promotion;
} siliconrt_decode_state_bindings_t;

siliconrt_status_t siliconrt_budget_create(
    uint64_t capacity_bytes,
    siliconrt_budget_t** out_budget);
void siliconrt_budget_destroy(siliconrt_budget_t* budget);
siliconrt_status_t siliconrt_budget_stats(
    const siliconrt_budget_t* budget,
    siliconrt_budget_stats_t* out_stats);

siliconrt_status_t siliconrt_arena_create(
    uint64_t capacity_bytes,
    siliconrt_arena_t** out_arena);
siliconrt_status_t siliconrt_arena_create_partitioned(
    uint64_t sequence_capacity_bytes,
    uint64_t constant_capacity_bytes,
    siliconrt_arena_t** out_arena);
void siliconrt_arena_destroy(siliconrt_arena_t* arena);
siliconrt_status_t siliconrt_arena_stats(
    const siliconrt_arena_t* arena,
    siliconrt_arena_stats_t* out_stats);
siliconrt_status_t siliconrt_arena_describe_backing_stores(
    const siliconrt_arena_t* arena,
    siliconrt_backing_store_layout_t* out_layout);

siliconrt_status_t siliconrt_prefix_create(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_descriptor_t* descriptor,
    siliconrt_prefix_handle_t** out_handle);
void siliconrt_prefix_destroy(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_prefix_handle_t* handle);

siliconrt_status_t siliconrt_prefix_compatible(
    const siliconrt_prefix_handle_t* handle,
    const char* model_key,
    const char* prefix_hash_hex,
    siliconrt_cache_mode_t cache_mode,
    bool* out_compatible);
siliconrt_status_t siliconrt_prefix_describe(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_prefix_descriptor_t* out_descriptor);
siliconrt_status_t siliconrt_prefix_describe_storage_handles(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_handle_layout_t* out_layout);
siliconrt_status_t siliconrt_prefix_describe_storage(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_layout_t* out_layout);

siliconrt_status_t siliconrt_decode_restore(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle,
    siliconrt_decode_state_t** out_state);
siliconrt_status_t siliconrt_decode_restore_borrowed(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle,
    siliconrt_decode_state_t** out_state);
siliconrt_status_t siliconrt_decode_state_describe(
    const siliconrt_decode_state_t* state,
    siliconrt_prefix_descriptor_t* out_descriptor);
siliconrt_status_t siliconrt_decode_state_describe_storage_handles(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_handle_layout_t* out_layout);
siliconrt_status_t siliconrt_decode_state_describe_storage(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_layout_t* out_layout);
siliconrt_status_t siliconrt_decode_state_describe_bindings(
    const siliconrt_decode_state_t* state,
    siliconrt_decode_state_bindings_t* out_bindings);
siliconrt_status_t siliconrt_decode_state_promote_sequence(
    siliconrt_decode_state_t* state);
siliconrt_status_t siliconrt_decode_state_set_residency(
    siliconrt_decode_state_t* state,
    uint32_t logical_token_count,
    uint32_t resident_token_count,
    uint64_t sequence_bytes);
siliconrt_status_t siliconrt_decode_state_set_residency_promoting(
    siliconrt_decode_state_t* state,
    uint32_t logical_token_count,
    uint32_t resident_token_count,
    uint64_t sequence_bytes);
void siliconrt_decode_state_destroy(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_decode_state_t* state);

siliconrt_status_t siliconrt_prefill_begin(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const char* model_key,
    siliconrt_prefill_handle_t** out_prefill);
void siliconrt_prefill_destroy(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_prefill_handle_t* prefill);

siliconrt_status_t siliconrt_prefill_finish_as_prefix(
    siliconrt_prefill_handle_t* prefill,
    const siliconrt_prefix_descriptor_t* descriptor,
    siliconrt_prefix_handle_t** out_handle);

#ifdef __cplusplus
}  // extern "C"
#endif
