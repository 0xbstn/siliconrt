#include "siliconrt/c_api.h"

#include <atomic>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>

#include "siliconrt/kv_arena.hpp"
#include "siliconrt/kv_budget.hpp"
#include "siliconrt/kv_types.hpp"
#include "siliconrt/partitioned_kv_arena.hpp"

using siliconrt::CacheMode;
using siliconrt::KvArena;
using siliconrt::KvBudget;
using siliconrt::PartitionedKvArena;
using siliconrt::PrefixHandle;
using siliconrt::ResidencyClass;

namespace {

std::atomic<std::uint64_t> g_next_backing_store_id{1};
std::atomic<std::uint64_t> g_next_storage_handle_id{1};

}  // namespace

struct siliconrt_budget {
  explicit siliconrt_budget(std::uint64_t capacity_bytes)
      : budget(capacity_bytes) {}

  bool reserve(std::uint64_t sequence_bytes, std::uint64_t constant_bytes) {
    const auto total_bytes = sequence_bytes + constant_bytes;
    if (!budget.reserve(total_bytes)) {
      return false;
    }
    sequence_reserved_bytes += sequence_bytes;
    constant_reserved_bytes += constant_bytes;
    return true;
  }

  bool commit_reserved(
      std::uint64_t sequence_bytes,
      std::uint64_t constant_bytes) {
    const auto total_bytes = sequence_bytes + constant_bytes;
    if (!budget.commit_reserved(total_bytes)) {
      return false;
    }
    if (sequence_bytes > sequence_reserved_bytes ||
        constant_bytes > constant_reserved_bytes) {
      return false;
    }
    sequence_reserved_bytes -= sequence_bytes;
    constant_reserved_bytes -= constant_bytes;
    sequence_committed_bytes += sequence_bytes;
    constant_committed_bytes += constant_bytes;
    return true;
  }

  bool release_reserved(
      std::uint64_t sequence_bytes,
      std::uint64_t constant_bytes) {
    const auto total_bytes = sequence_bytes + constant_bytes;
    if (!budget.release_reserved(total_bytes)) {
      return false;
    }
    if (sequence_bytes > sequence_reserved_bytes ||
        constant_bytes > constant_reserved_bytes) {
      return false;
    }
    sequence_reserved_bytes -= sequence_bytes;
    constant_reserved_bytes -= constant_bytes;
    return true;
  }

  bool release_committed(
      std::uint64_t sequence_bytes,
      std::uint64_t constant_bytes) {
    const auto total_bytes = sequence_bytes + constant_bytes;
    if (!budget.release_committed(total_bytes)) {
      return false;
    }
    if (sequence_bytes > sequence_committed_bytes ||
        constant_bytes > constant_committed_bytes) {
      return false;
    }
    sequence_committed_bytes -= sequence_bytes;
    constant_committed_bytes -= constant_bytes;
    return true;
  }

  KvBudget budget;
  std::uint64_t sequence_reserved_bytes = 0;
  std::uint64_t constant_reserved_bytes = 0;
  std::uint64_t sequence_committed_bytes = 0;
  std::uint64_t constant_committed_bytes = 0;
};

struct siliconrt_arena {
  explicit siliconrt_arena(std::uint64_t capacity_bytes)
      : unified_arena(capacity_bytes),
        unified_store_id(g_next_backing_store_id.fetch_add(1)) {}

  siliconrt_arena(
      std::uint64_t sequence_capacity_bytes,
      std::uint64_t constant_capacity_bytes)
      : partitioned_arena(
            std::in_place, sequence_capacity_bytes, constant_capacity_bytes),
        sequence_store_id(g_next_backing_store_id.fetch_add(1)),
        constant_store_id(g_next_backing_store_id.fetch_add(1)) {}

  [[nodiscard]] bool is_partitioned() const {
    return partitioned_arena.has_value();
  }

  [[nodiscard]] std::uint64_t capacity_bytes() const {
    if (partitioned_arena.has_value()) {
      return partitioned_arena->total_capacity_bytes();
    }
    return unified_arena->capacity_bytes();
  }

  std::optional<siliconrt::KvSpan> allocate(
      std::uint64_t capacity_bytes,
      std::uint32_t token_capacity,
      ResidencyClass residency_class) {
    std::optional<siliconrt::KvSpan> span;
    if (partitioned_arena.has_value()) {
      span = partitioned_arena->allocate(
          capacity_bytes, token_capacity, residency_class);
    } else {
      span = unified_arena->allocate(capacity_bytes, token_capacity, residency_class);
    }
    if (span.has_value()) {
      storage_handle_ids.emplace(
          span->span_id, g_next_storage_handle_id.fetch_add(1));
    }
    return span;
  }

  bool commit(
      std::uint64_t span_id,
      std::uint64_t used_bytes,
      std::uint32_t token_count) {
    if (partitioned_arena.has_value()) {
      return partitioned_arena->commit(span_id, used_bytes, token_count);
    }
    return unified_arena->commit(span_id, used_bytes, token_count);
  }

  bool release(std::uint64_t span_id) {
    storage_handle_ids.erase(span_id);
    if (partitioned_arena.has_value()) {
      return partitioned_arena->release(span_id);
    }
    return unified_arena->release(span_id);
  }

  [[nodiscard]] std::optional<siliconrt::KvSpan> get(std::uint64_t span_id) const {
    if (partitioned_arena.has_value()) {
      return partitioned_arena->get(span_id);
    }
    return unified_arena->get(span_id);
  }

  struct SpanStorageView {
    siliconrt::KvSpan span = {};
    std::uint64_t storage_handle_id = 0;
    siliconrt_backing_store_kind_t backing_store_kind =
        SILICONRT_BACKING_STORE_KIND_UNKNOWN;
    std::uint64_t backing_store_id = 0;
    std::uint64_t backing_store_offset_bytes = 0;
  };

  [[nodiscard]] std::optional<SpanStorageView> describe(std::uint64_t span_id) const {
    const auto handle_it = storage_handle_ids.find(span_id);
    if (handle_it == storage_handle_ids.end()) {
      return std::nullopt;
    }
    if (partitioned_arena.has_value()) {
      const auto view = partitioned_arena->describe(span_id);
      if (!view.has_value()) {
        return std::nullopt;
      }
      SpanStorageView out;
      out.span = view->global_span;
      out.storage_handle_id = handle_it->second;
      out.backing_store_offset_bytes = view->backing_offset_bytes;
      if (view->global_span.residency_class == ResidencyClass::kSequenceGrowing) {
        out.backing_store_kind = SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL;
        out.backing_store_id = sequence_store_id;
      } else if (
          view->global_span.residency_class == ResidencyClass::kConstantState) {
        out.backing_store_kind = SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL;
        out.backing_store_id = constant_store_id;
      }
      return out;
    }

    const auto span = unified_arena->get(span_id);
    if (!span.has_value()) {
      return std::nullopt;
    }
    SpanStorageView out;
    out.span = *span;
    out.storage_handle_id = handle_it->second;
    out.backing_store_kind = SILICONRT_BACKING_STORE_KIND_HOST_UNIFIED;
    out.backing_store_id = unified_store_id;
    out.backing_store_offset_bytes = span->offset_bytes;
    return out;
  }

  [[nodiscard]] siliconrt_arena_stats_t stats() const {
    siliconrt_arena_stats_t out = {};
    if (partitioned_arena.has_value()) {
      const auto stats = partitioned_arena->stats();
      out.capacity_bytes = stats.total_capacity_bytes;
      out.free_bytes = stats.total_free_bytes;
      out.allocated_capacity_bytes = stats.total_allocated_capacity_bytes;
      out.used_bytes = stats.total_used_bytes;
      out.allocated_span_count = stats.total_allocated_span_count;
      out.is_partitioned = true;
      out.sequence_capacity_bytes = stats.sequence_pool.capacity_bytes;
      out.constant_capacity_bytes = stats.constant_pool.capacity_bytes;
      out.sequence_used_bytes = stats.sequence_pool.used_bytes;
      out.constant_used_bytes = stats.constant_pool.used_bytes;
      out.largest_free_range_bytes = stats.sequence_pool.largest_free_range_bytes >
              stats.constant_pool.largest_free_range_bytes
          ? stats.sequence_pool.largest_free_range_bytes
          : stats.constant_pool.largest_free_range_bytes;
      return out;
    }

    const auto stats = unified_arena->stats();
    out.capacity_bytes = stats.capacity_bytes;
    out.free_bytes = stats.free_bytes;
    out.largest_free_range_bytes = stats.largest_free_range_bytes;
    out.allocated_capacity_bytes = stats.allocated_capacity_bytes;
    out.used_bytes = stats.used_bytes;
    out.allocated_span_count = stats.allocated_span_count;
    out.is_partitioned = false;
    out.sequence_capacity_bytes = stats.capacity_bytes;
    out.constant_capacity_bytes = 0;
    out.sequence_used_bytes = stats.used_bytes;
    out.constant_used_bytes = 0;
    return out;
  }

  void describe_backing_stores(siliconrt_backing_store_layout_t* out_layout) const {
    out_layout->sequence = {};
    out_layout->constant_state = {};

    if (partitioned_arena.has_value()) {
      out_layout->sequence.present = true;
      out_layout->sequence.kind = SILICONRT_BACKING_STORE_KIND_HOST_SEQUENCE_POOL;
      out_layout->sequence.backing_store_id = sequence_store_id;
      out_layout->sequence.capacity_bytes =
          partitioned_arena->sequence_capacity_bytes();
      out_layout->sequence.global_base_offset_bytes = 0;

      out_layout->constant_state.present = true;
      out_layout->constant_state.kind =
          SILICONRT_BACKING_STORE_KIND_HOST_CONSTANT_POOL;
      out_layout->constant_state.backing_store_id = constant_store_id;
      out_layout->constant_state.capacity_bytes =
          partitioned_arena->constant_capacity_bytes();
      out_layout->constant_state.global_base_offset_bytes =
          partitioned_arena->sequence_capacity_bytes();
      return;
    }

    out_layout->sequence.present = true;
    out_layout->sequence.kind = SILICONRT_BACKING_STORE_KIND_HOST_UNIFIED;
    out_layout->sequence.backing_store_id = unified_store_id;
    out_layout->sequence.capacity_bytes = unified_arena->capacity_bytes();
    out_layout->sequence.global_base_offset_bytes = 0;

    out_layout->constant_state = out_layout->sequence;
  }

  std::optional<KvArena> unified_arena;
  std::optional<PartitionedKvArena> partitioned_arena;
  std::unordered_map<std::uint64_t, std::uint64_t> storage_handle_ids;
  std::uint64_t unified_store_id = 0;
  std::uint64_t sequence_store_id = 0;
  std::uint64_t constant_store_id = 0;
};

struct siliconrt_prefix_handle {
  siliconrt_arena_t* owner_arena = nullptr;
  siliconrt_budget_t* owner_budget = nullptr;
  PrefixHandle handle;
  std::uint64_t active_borrowed_decode_count = 0;
  bool destroy_requested = false;
};

struct siliconrt_decode_state {
  siliconrt_arena_t* owner_arena = nullptr;
  siliconrt_budget_t* owner_budget = nullptr;
  siliconrt_prefix_handle_t* borrowed_prefix = nullptr;
  std::string model_key;
  std::string prefix_hash_hex;
  CacheMode cache_mode = CacheMode::kUnknown;
  std::uint64_t sequence_span_id = 0;
  std::uint64_t constant_span_id = 0;
  std::uint64_t borrowed_sequence_span_id = 0;
  std::uint64_t borrowed_constant_span_id = 0;
  bool owns_sequence = true;
  bool owns_constant = true;
  std::uint32_t logical_token_count = 0;
  std::uint32_t resident_token_count = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
};

struct siliconrt_prefill_handle {
  siliconrt_arena_t* arena = nullptr;
  siliconrt_budget_t* budget = nullptr;
  std::string model_key;
  bool finished = false;
};

namespace {

std::atomic<std::uint64_t> g_next_prefix_handle_id{1};

struct AllocationRecord {
  std::uint64_t sequence_span_id = 0;
  std::uint64_t constant_span_id = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
};

struct DecodeStateOwnedAllocation {
  std::uint64_t sequence_span_id = 0;
  std::uint64_t constant_span_id = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
};

CacheMode from_api_mode(siliconrt_cache_mode_t mode) {
  switch (mode) {
    case SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS:
      return CacheMode::kBoundedContiguous;
    case SILICONRT_CACHE_MODE_PAGED:
      return CacheMode::kPaged;
    default:
      return CacheMode::kUnknown;
  }
}

siliconrt_cache_mode_t to_api_mode(CacheMode mode) {
  switch (mode) {
    case CacheMode::kBoundedContiguous:
      return SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS;
    case CacheMode::kPaged:
      return SILICONRT_CACHE_MODE_PAGED;
    default:
      return SILICONRT_CACHE_MODE_UNKNOWN;
  }
}

siliconrt_storage_kind_t to_api_storage_kind(ResidencyClass residency_class) {
  switch (residency_class) {
    case ResidencyClass::kSequenceGrowing:
      return SILICONRT_STORAGE_KIND_SEQUENCE;
    case ResidencyClass::kConstantState:
      return SILICONRT_STORAGE_KIND_CONSTANT;
    default:
      return SILICONRT_STORAGE_KIND_UNKNOWN;
  }
}

siliconrt_storage_ownership_t to_api_storage_ownership(bool owns, bool borrows) {
  if (owns) {
    return SILICONRT_STORAGE_OWNERSHIP_OWNED;
  }
  if (borrows) {
    return SILICONRT_STORAGE_OWNERSHIP_BORROWED;
  }
  return SILICONRT_STORAGE_OWNERSHIP_UNKNOWN;
}

void release_allocation(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const AllocationRecord& allocation) {
  if (arena != nullptr) {
    if (allocation.sequence_span_id != 0) {
      arena->release(allocation.sequence_span_id);
    }
    if (allocation.constant_span_id != 0) {
      arena->release(allocation.constant_span_id);
    }
  }

  if (budget != nullptr) {
    budget->release_committed(
        allocation.sequence_bytes, allocation.constant_bytes);
  }
}

void release_decode_owned_allocation(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const DecodeStateOwnedAllocation& allocation) {
  if (arena != nullptr) {
    if (allocation.sequence_span_id != 0) {
      arena->release(allocation.sequence_span_id);
    }
    if (allocation.constant_span_id != 0) {
      arena->release(allocation.constant_span_id);
    }
  }

  if (budget != nullptr) {
    budget->release_committed(
        allocation.sequence_bytes, allocation.constant_bytes);
  }
}

DecodeStateOwnedAllocation owned_allocation_from_state(
    const siliconrt_decode_state_t* state) {
  DecodeStateOwnedAllocation allocation;
  if (state->owns_sequence) {
    allocation.sequence_span_id = state->sequence_span_id;
    allocation.sequence_bytes = state->sequence_bytes;
  }
  if (state->owns_constant) {
    allocation.constant_span_id = state->constant_span_id;
    allocation.constant_bytes = state->constant_bytes;
  }
  return allocation;
}

void maybe_finalize_prefix_handle(siliconrt_prefix_handle_t* handle) {
  if (handle == nullptr || !handle->destroy_requested ||
      handle->active_borrowed_decode_count != 0) {
    return;
  }

  AllocationRecord allocation;
  allocation.sequence_span_id = handle->handle.sequence_span_id;
  allocation.constant_span_id = handle->handle.constant_span_id;
  allocation.sequence_bytes = handle->handle.sequence_bytes;
  allocation.constant_bytes = handle->handle.constant_bytes;
  release_allocation(handle->owner_arena, handle->owner_budget, allocation);
  delete handle;
}

bool prefix_handle_available_for_restore(const siliconrt_prefix_handle_t* handle) {
  return handle != nullptr && !handle->destroy_requested;
}

siliconrt_status_t validate_descriptor(
    const siliconrt_prefix_descriptor_t* descriptor) {
  if (descriptor == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor->model_key == nullptr || descriptor->prefix_hash_hex == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  const auto cache_mode = from_api_mode(descriptor->cache_mode);
  if (cache_mode == CacheMode::kUnknown) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (cache_mode != CacheMode::kBoundedContiguous) {
    return SILICONRT_STATUS_UNIMPLEMENTED;
  }
  if (descriptor->sequence_bytes > 0 &&
      descriptor->logical_token_count == 0) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor->resident_token_count > descriptor->logical_token_count) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (descriptor->sequence_bytes > 0 &&
      descriptor->resident_token_count == 0) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t allocate_prefix_storage(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    std::uint32_t resident_token_count,
    std::uint64_t sequence_bytes,
    std::uint64_t constant_bytes,
    AllocationRecord* out_allocation) {
  if (arena == nullptr || budget == nullptr || out_allocation == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  const auto total_bytes = sequence_bytes + constant_bytes;
  if (!budget->reserve(sequence_bytes, constant_bytes)) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }

  AllocationRecord allocation;
  allocation.sequence_bytes = sequence_bytes;
  allocation.constant_bytes = constant_bytes;

  if (sequence_bytes > 0) {
    auto sequence_span = arena->allocate(
        sequence_bytes,
        resident_token_count,
        ResidencyClass::kSequenceGrowing);
    if (!sequence_span.has_value()) {
      budget->release_reserved(sequence_bytes, constant_bytes);
      return SILICONRT_STATUS_OUT_OF_MEMORY;
    }
    if (!arena->commit(
            sequence_span->span_id, sequence_bytes, resident_token_count)) {
      arena->release(sequence_span->span_id);
      budget->release_reserved(sequence_bytes, constant_bytes);
      return SILICONRT_STATUS_INVALID_ARGUMENT;
    }
    allocation.sequence_span_id = sequence_span->span_id;
  }

  if (constant_bytes > 0) {
    auto constant_span = arena->allocate(
        constant_bytes,
        0,
        ResidencyClass::kConstantState);
    if (!constant_span.has_value()) {
      if (allocation.sequence_span_id != 0) {
        arena->release(allocation.sequence_span_id);
      }
      budget->release_reserved(sequence_bytes, constant_bytes);
      return SILICONRT_STATUS_OUT_OF_MEMORY;
    }
    if (!arena->commit(constant_span->span_id, constant_bytes, 0)) {
      arena->release(constant_span->span_id);
      if (allocation.sequence_span_id != 0) {
        arena->release(allocation.sequence_span_id);
      }
      budget->release_reserved(sequence_bytes, constant_bytes);
      return SILICONRT_STATUS_INVALID_ARGUMENT;
    }
    allocation.constant_span_id = constant_span->span_id;
  }

  if (!budget->commit_reserved(sequence_bytes, constant_bytes)) {
    if (allocation.sequence_span_id != 0) {
      arena->release(allocation.sequence_span_id);
    }
    if (allocation.constant_span_id != 0) {
      arena->release(allocation.constant_span_id);
    }
    budget->release_reserved(sequence_bytes, constant_bytes);
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  *out_allocation = allocation;
  return SILICONRT_STATUS_OK;
}

void fill_descriptor_from_prefix_handle(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_prefix_descriptor_t* out_descriptor) {
  out_descriptor->model_key = handle->handle.model_key.c_str();
  out_descriptor->prefix_hash_hex = handle->handle.prefix_hash_hex.c_str();
  out_descriptor->logical_token_count = handle->handle.logical_token_count;
  out_descriptor->resident_token_count = handle->handle.resident_token_count;
  out_descriptor->sequence_bytes = handle->handle.sequence_bytes;
  out_descriptor->constant_bytes = handle->handle.constant_bytes;
  out_descriptor->cache_mode = to_api_mode(handle->handle.cache_mode);
}

void fill_descriptor_from_decode_state(
    const siliconrt_decode_state_t* state,
    siliconrt_prefix_descriptor_t* out_descriptor) {
  out_descriptor->model_key = state->model_key.c_str();
  out_descriptor->prefix_hash_hex = state->prefix_hash_hex.c_str();
  out_descriptor->logical_token_count = state->logical_token_count;
  out_descriptor->resident_token_count = state->resident_token_count;
  out_descriptor->sequence_bytes = state->sequence_bytes;
  out_descriptor->constant_bytes = state->constant_bytes;
  out_descriptor->cache_mode = to_api_mode(state->cache_mode);
}

void fill_bindings_from_decode_state(
    const siliconrt_decode_state_t* state,
    siliconrt_decode_state_bindings_t* out_bindings) {
  out_bindings->owns_sequence = state->owns_sequence;
  out_bindings->borrows_sequence = state->borrowed_sequence_span_id != 0;
  out_bindings->owns_constant = state->owns_constant;
  out_bindings->borrows_constant = state->borrowed_constant_span_id != 0;
  out_bindings->requires_sequence_promotion =
      !state->owns_sequence && state->borrowed_sequence_span_id != 0;
}

void fill_storage_binding_empty(
    siliconrt_storage_kind_t storage_kind,
    siliconrt_storage_binding_t* out_binding) {
  out_binding->present = false;
  out_binding->storage_kind = storage_kind;
  out_binding->ownership = SILICONRT_STORAGE_OWNERSHIP_UNKNOWN;
  out_binding->backing_store_kind = SILICONRT_BACKING_STORE_KIND_UNKNOWN;
  out_binding->backing_store_id = 0;
  out_binding->backing_store_offset_bytes = 0;
  out_binding->span_id = 0;
  out_binding->offset_bytes = 0;
  out_binding->capacity_bytes = 0;
  out_binding->used_bytes = 0;
  out_binding->token_capacity = 0;
  out_binding->token_count = 0;
}

void fill_storage_binding_from_span(
    const std::optional<siliconrt_arena::SpanStorageView>& view,
    siliconrt_storage_ownership_t ownership,
    siliconrt_storage_binding_t* out_binding) {
  if (!view.has_value()) {
    fill_storage_binding_empty(SILICONRT_STORAGE_KIND_UNKNOWN, out_binding);
    out_binding->ownership = ownership;
    return;
  }

  out_binding->present = true;
  out_binding->storage_kind = to_api_storage_kind(view->span.residency_class);
  out_binding->ownership = ownership;
  out_binding->backing_store_kind = view->backing_store_kind;
  out_binding->backing_store_id = view->backing_store_id;
  out_binding->backing_store_offset_bytes = view->backing_store_offset_bytes;
  out_binding->span_id = view->span.span_id;
  out_binding->offset_bytes = view->span.offset_bytes;
  out_binding->capacity_bytes = view->span.capacity_bytes;
  out_binding->used_bytes = view->span.used_bytes;
  out_binding->token_capacity = view->span.token_capacity;
  out_binding->token_count = view->span.token_count;
}

void fill_storage_handle_descriptor_empty(
    siliconrt_storage_kind_t storage_kind,
    siliconrt_storage_handle_descriptor_t* out_descriptor) {
  out_descriptor->present = false;
  out_descriptor->storage_kind = storage_kind;
  out_descriptor->storage_handle_id = 0;
  out_descriptor->backing_store_kind = SILICONRT_BACKING_STORE_KIND_UNKNOWN;
  out_descriptor->backing_store_id = 0;
  out_descriptor->backing_store_offset_bytes = 0;
  out_descriptor->capacity_bytes = 0;
  out_descriptor->token_capacity = 0;
}

void fill_storage_handle_descriptor_from_span(
    const std::optional<siliconrt_arena::SpanStorageView>& view,
    siliconrt_storage_handle_descriptor_t* out_descriptor) {
  if (!view.has_value()) {
    fill_storage_handle_descriptor_empty(
        SILICONRT_STORAGE_KIND_UNKNOWN, out_descriptor);
    return;
  }

  out_descriptor->present = true;
  out_descriptor->storage_kind = to_api_storage_kind(view->span.residency_class);
  out_descriptor->storage_handle_id = view->storage_handle_id;
  out_descriptor->backing_store_kind = view->backing_store_kind;
  out_descriptor->backing_store_id = view->backing_store_id;
  out_descriptor->backing_store_offset_bytes = view->backing_store_offset_bytes;
  out_descriptor->capacity_bytes = view->span.capacity_bytes;
  out_descriptor->token_capacity = view->span.token_capacity;
}

siliconrt_status_t fill_storage_layout_from_prefix_handle(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_layout_t* out_layout) {
  if (handle == nullptr || out_layout == nullptr || handle->owner_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  if (handle->handle.sequence_span_id != 0) {
    fill_storage_binding_from_span(
        handle->owner_arena->describe(handle->handle.sequence_span_id),
        SILICONRT_STORAGE_OWNERSHIP_OWNED,
        &out_layout->sequence);
    if (!out_layout->sequence.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_binding_empty(
        SILICONRT_STORAGE_KIND_SEQUENCE, &out_layout->sequence);
  }

  if (handle->handle.constant_span_id != 0) {
    fill_storage_binding_from_span(
        handle->owner_arena->describe(handle->handle.constant_span_id),
        SILICONRT_STORAGE_OWNERSHIP_OWNED,
        &out_layout->constant_state);
    if (!out_layout->constant_state.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_binding_empty(
        SILICONRT_STORAGE_KIND_CONSTANT, &out_layout->constant_state);
  }

  return SILICONRT_STATUS_OK;
}

siliconrt_status_t fill_storage_layout_from_decode_state(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_layout_t* out_layout) {
  if (state == nullptr || out_layout == nullptr || state->owner_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  const auto sequence_span_id = state->sequence_span_id;
  const auto sequence_ownership = to_api_storage_ownership(
      state->owns_sequence,
      state->borrowed_sequence_span_id != 0);
  if (sequence_span_id != 0) {
    fill_storage_binding_from_span(
        state->owner_arena->describe(sequence_span_id),
        sequence_ownership,
        &out_layout->sequence);
    if (!out_layout->sequence.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_binding_empty(
        SILICONRT_STORAGE_KIND_SEQUENCE, &out_layout->sequence);
    out_layout->sequence.ownership = sequence_ownership;
  }

  const auto constant_span_id = state->constant_span_id;
  const auto constant_ownership = to_api_storage_ownership(
      state->owns_constant,
      state->borrowed_constant_span_id != 0);
  if (constant_span_id != 0) {
    fill_storage_binding_from_span(
        state->owner_arena->describe(constant_span_id),
        constant_ownership,
        &out_layout->constant_state);
    if (!out_layout->constant_state.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_binding_empty(
        SILICONRT_STORAGE_KIND_CONSTANT, &out_layout->constant_state);
    out_layout->constant_state.ownership = constant_ownership;
  }

  return SILICONRT_STATUS_OK;
}

siliconrt_status_t fill_storage_handle_layout_from_prefix_handle(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_handle_layout_t* out_layout) {
  if (handle == nullptr || out_layout == nullptr || handle->owner_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  if (handle->handle.sequence_span_id != 0) {
    fill_storage_handle_descriptor_from_span(
        handle->owner_arena->describe(handle->handle.sequence_span_id),
        &out_layout->sequence);
    if (!out_layout->sequence.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_handle_descriptor_empty(
        SILICONRT_STORAGE_KIND_SEQUENCE, &out_layout->sequence);
  }

  if (handle->handle.constant_span_id != 0) {
    fill_storage_handle_descriptor_from_span(
        handle->owner_arena->describe(handle->handle.constant_span_id),
        &out_layout->constant_state);
    if (!out_layout->constant_state.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_handle_descriptor_empty(
        SILICONRT_STORAGE_KIND_CONSTANT, &out_layout->constant_state);
  }

  return SILICONRT_STATUS_OK;
}

siliconrt_status_t fill_storage_handle_layout_from_decode_state(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_handle_layout_t* out_layout) {
  if (state == nullptr || out_layout == nullptr || state->owner_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  if (state->sequence_span_id != 0) {
    fill_storage_handle_descriptor_from_span(
        state->owner_arena->describe(state->sequence_span_id),
        &out_layout->sequence);
    if (!out_layout->sequence.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_handle_descriptor_empty(
        SILICONRT_STORAGE_KIND_SEQUENCE, &out_layout->sequence);
  }

  if (state->constant_span_id != 0) {
    fill_storage_handle_descriptor_from_span(
        state->owner_arena->describe(state->constant_span_id),
        &out_layout->constant_state);
    if (!out_layout->constant_state.present) {
      return SILICONRT_STATUS_NOT_FOUND;
    }
  } else {
    fill_storage_handle_descriptor_empty(
        SILICONRT_STORAGE_KIND_CONSTANT, &out_layout->constant_state);
  }

  return SILICONRT_STATUS_OK;
}

siliconrt_status_t promote_decode_sequence_impl(
    siliconrt_decode_state_t* state) {
  if (state == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (state->owns_sequence) {
    return SILICONRT_STATUS_OK;
  }
  if (state->borrowed_sequence_span_id == 0 || state->sequence_bytes == 0) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  AllocationRecord allocation;
  const auto allocation_status = allocate_prefix_storage(
      state->owner_arena,
      state->owner_budget,
      state->resident_token_count,
      state->sequence_bytes,
      0,
      &allocation);
  if (allocation_status != SILICONRT_STATUS_OK) {
    return allocation_status;
  }

  state->sequence_span_id = allocation.sequence_span_id;
  state->owns_sequence = true;
  state->borrowed_sequence_span_id = 0;
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t set_decode_state_residency_impl(
    siliconrt_decode_state_t* state,
    uint32_t logical_token_count,
    uint32_t resident_token_count,
    uint64_t sequence_bytes,
    bool allow_promotion) {
  if (state == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (resident_token_count > logical_token_count) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (sequence_bytes > 0 && resident_token_count == 0) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (!state->owns_sequence) {
    const auto no_op =
        logical_token_count == state->logical_token_count &&
        resident_token_count == state->resident_token_count &&
        sequence_bytes == state->sequence_bytes;
    if (no_op) {
      return SILICONRT_STATUS_OK;
    }
    if (!allow_promotion) {
      return SILICONRT_STATUS_INVALID_ARGUMENT;
    }
    const auto promote_status = promote_decode_sequence_impl(state);
    if (promote_status != SILICONRT_STATUS_OK) {
      return promote_status;
    }
  }

  const auto span = state->owner_arena->get(state->sequence_span_id);
  if (!span.has_value()) {
    return SILICONRT_STATUS_NOT_FOUND;
  }
  if (sequence_bytes > span->capacity_bytes ||
      resident_token_count > span->token_capacity) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
  if (!state->owner_arena->commit(
          state->sequence_span_id, sequence_bytes, resident_token_count)) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }

  state->logical_token_count = logical_token_count;
  state->resident_token_count = resident_token_count;
  state->sequence_bytes = sequence_bytes;
  return SILICONRT_STATUS_OK;
}

}  // namespace

extern "C" {

siliconrt_status_t siliconrt_budget_create(
    uint64_t capacity_bytes,
    siliconrt_budget_t** out_budget) {
  if (out_budget == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_budget = nullptr;
  try {
    *out_budget = new siliconrt_budget_t(capacity_bytes);
  } catch (const std::bad_alloc&) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
  return SILICONRT_STATUS_OK;
}

void siliconrt_budget_destroy(siliconrt_budget_t* budget) {
  delete budget;
}

siliconrt_status_t siliconrt_budget_stats(
    const siliconrt_budget_t* budget,
    siliconrt_budget_stats_t* out_stats) {
  if (budget == nullptr || out_stats == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  out_stats->capacity_bytes = budget->budget.capacity_bytes();
  out_stats->reserved_bytes = budget->budget.reserved_bytes();
  out_stats->committed_bytes = budget->budget.committed_bytes();
  out_stats->sequence_reserved_bytes = budget->sequence_reserved_bytes;
  out_stats->constant_reserved_bytes = budget->constant_reserved_bytes;
  out_stats->sequence_committed_bytes = budget->sequence_committed_bytes;
  out_stats->constant_committed_bytes = budget->constant_committed_bytes;
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_arena_create(
    uint64_t capacity_bytes,
    siliconrt_arena_t** out_arena) {
  if (out_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_arena = nullptr;
  try {
    *out_arena = new siliconrt_arena_t(capacity_bytes);
  } catch (const std::bad_alloc&) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_arena_create_partitioned(
    uint64_t sequence_capacity_bytes,
    uint64_t constant_capacity_bytes,
    siliconrt_arena_t** out_arena) {
  if (out_arena == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_arena = nullptr;
  try {
    *out_arena =
        new siliconrt_arena_t(sequence_capacity_bytes, constant_capacity_bytes);
  } catch (const std::bad_alloc&) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
  return SILICONRT_STATUS_OK;
}

void siliconrt_arena_destroy(siliconrt_arena_t* arena) {
  delete arena;
}

siliconrt_status_t siliconrt_arena_stats(
    const siliconrt_arena_t* arena,
    siliconrt_arena_stats_t* out_stats) {
  if (arena == nullptr || out_stats == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_stats = arena->stats();
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_arena_describe_backing_stores(
    const siliconrt_arena_t* arena,
    siliconrt_backing_store_layout_t* out_layout) {
  if (arena == nullptr || out_layout == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  arena->describe_backing_stores(out_layout);
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_prefix_create(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_descriptor_t* descriptor,
    siliconrt_prefix_handle_t** out_handle) {
  if (arena == nullptr || budget == nullptr || out_handle == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_handle = nullptr;
  const auto descriptor_status = validate_descriptor(descriptor);
  if (descriptor_status != SILICONRT_STATUS_OK) {
    return descriptor_status;
  }

  AllocationRecord allocation;
  const auto allocation_status = allocate_prefix_storage(
      arena,
      budget,
      descriptor->resident_token_count,
      descriptor->sequence_bytes,
      descriptor->constant_bytes,
      &allocation);
  if (allocation_status != SILICONRT_STATUS_OK) {
    return allocation_status;
  }

  try {
    auto* handle = new siliconrt_prefix_handle_t();
    handle->owner_arena = arena;
    handle->owner_budget = budget;
    handle->handle.handle_id = g_next_prefix_handle_id.fetch_add(1);
    handle->handle.model_key = descriptor->model_key;
    handle->handle.prefix_hash_hex = descriptor->prefix_hash_hex;
    handle->handle.cache_mode = from_api_mode(descriptor->cache_mode);
    handle->handle.sequence_span_id = allocation.sequence_span_id;
    handle->handle.constant_span_id = allocation.constant_span_id;
    handle->handle.logical_token_count = descriptor->logical_token_count;
    handle->handle.resident_token_count = descriptor->resident_token_count;
    handle->handle.sequence_bytes = descriptor->sequence_bytes;
    handle->handle.constant_bytes = descriptor->constant_bytes;
    *out_handle = handle;
    return SILICONRT_STATUS_OK;
  } catch (const std::bad_alloc&) {
    release_allocation(arena, budget, allocation);
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
}

void siliconrt_prefix_destroy(
    siliconrt_arena_t*,
    siliconrt_budget_t*,
    siliconrt_prefix_handle_t* handle) {
  if (handle == nullptr) {
    return;
  }
  handle->destroy_requested = true;
  maybe_finalize_prefix_handle(handle);
}

siliconrt_status_t siliconrt_prefix_compatible(
    const siliconrt_prefix_handle_t* handle,
    const char* model_key,
    const char* prefix_hash_hex,
    siliconrt_cache_mode_t cache_mode,
    bool* out_compatible) {
  if (handle == nullptr || model_key == nullptr || prefix_hash_hex == nullptr ||
      out_compatible == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_compatible = handle->handle.compatible_with(
      model_key, prefix_hash_hex, from_api_mode(cache_mode));
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_prefix_describe(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_prefix_descriptor_t* out_descriptor) {
  if (handle == nullptr || out_descriptor == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  fill_descriptor_from_prefix_handle(handle, out_descriptor);
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_prefix_describe_storage_handles(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_handle_layout_t* out_layout) {
  return fill_storage_handle_layout_from_prefix_handle(handle, out_layout);
}

siliconrt_status_t siliconrt_prefix_describe_storage(
    const siliconrt_prefix_handle_t* handle,
    siliconrt_storage_layout_t* out_layout) {
  return fill_storage_layout_from_prefix_handle(handle, out_layout);
}

siliconrt_status_t siliconrt_decode_restore(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle,
    siliconrt_decode_state_t** out_state) {
  if (arena == nullptr || budget == nullptr || handle == nullptr ||
      out_state == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (!prefix_handle_available_for_restore(handle)) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_state = nullptr;

  AllocationRecord allocation;
  const auto allocation_status = allocate_prefix_storage(
      arena,
      budget,
      handle->handle.resident_token_count,
      handle->handle.sequence_bytes,
      handle->handle.constant_bytes,
      &allocation);
  if (allocation_status != SILICONRT_STATUS_OK) {
    return allocation_status;
  }

  try {
    auto* state = new siliconrt_decode_state_t();
    state->owner_arena = arena;
    state->owner_budget = budget;
    state->borrowed_prefix = nullptr;
    state->model_key = handle->handle.model_key;
    state->prefix_hash_hex = handle->handle.prefix_hash_hex;
    state->cache_mode = handle->handle.cache_mode;
    state->sequence_span_id = allocation.sequence_span_id;
    state->constant_span_id = allocation.constant_span_id;
    state->owns_sequence = true;
    state->owns_constant = true;
    state->logical_token_count = handle->handle.logical_token_count;
    state->resident_token_count = handle->handle.resident_token_count;
    state->sequence_bytes = handle->handle.sequence_bytes;
    state->constant_bytes = handle->handle.constant_bytes;
    *out_state = state;
    return SILICONRT_STATUS_OK;
  } catch (const std::bad_alloc&) {
    release_allocation(arena, budget, allocation);
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
}

siliconrt_status_t siliconrt_decode_restore_borrowed(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle,
    siliconrt_decode_state_t** out_state) {
  if (arena == nullptr || budget == nullptr || handle == nullptr ||
      out_state == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  if (!prefix_handle_available_for_restore(handle)) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_state = nullptr;

  try {
    auto* state = new siliconrt_decode_state_t();
    state->owner_arena = arena;
    state->owner_budget = budget;
    state->borrowed_prefix =
        const_cast<siliconrt_prefix_handle_t*>(handle);
    state->model_key = handle->handle.model_key;
    state->prefix_hash_hex = handle->handle.prefix_hash_hex;
    state->cache_mode = handle->handle.cache_mode;
    state->sequence_span_id = handle->handle.sequence_span_id;
    state->constant_span_id = handle->handle.constant_span_id;
    state->borrowed_sequence_span_id = handle->handle.sequence_span_id;
    state->borrowed_constant_span_id = handle->handle.constant_span_id;
    state->owns_sequence = false;
    state->owns_constant = false;
    state->logical_token_count = handle->handle.logical_token_count;
    state->resident_token_count = handle->handle.resident_token_count;
    state->sequence_bytes = handle->handle.sequence_bytes;
    state->constant_bytes = handle->handle.constant_bytes;
    state->borrowed_prefix->active_borrowed_decode_count += 1;
    *out_state = state;
    return SILICONRT_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
}

siliconrt_status_t siliconrt_decode_state_describe(
    const siliconrt_decode_state_t* state,
    siliconrt_prefix_descriptor_t* out_descriptor) {
  if (state == nullptr || out_descriptor == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  fill_descriptor_from_decode_state(state, out_descriptor);
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_decode_state_describe_storage_handles(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_handle_layout_t* out_layout) {
  return fill_storage_handle_layout_from_decode_state(state, out_layout);
}

siliconrt_status_t siliconrt_decode_state_describe_storage(
    const siliconrt_decode_state_t* state,
    siliconrt_storage_layout_t* out_layout) {
  return fill_storage_layout_from_decode_state(state, out_layout);
}

siliconrt_status_t siliconrt_decode_state_describe_bindings(
    const siliconrt_decode_state_t* state,
    siliconrt_decode_state_bindings_t* out_bindings) {
  if (state == nullptr || out_bindings == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  fill_bindings_from_decode_state(state, out_bindings);
  return SILICONRT_STATUS_OK;
}

siliconrt_status_t siliconrt_decode_state_promote_sequence(
    siliconrt_decode_state_t* state) {
  return promote_decode_sequence_impl(state);
}

siliconrt_status_t siliconrt_decode_state_set_residency(
    siliconrt_decode_state_t* state,
    uint32_t logical_token_count,
    uint32_t resident_token_count,
    uint64_t sequence_bytes) {
  return set_decode_state_residency_impl(
      state,
      logical_token_count,
      resident_token_count,
      sequence_bytes,
      false);
}

siliconrt_status_t siliconrt_decode_state_set_residency_promoting(
    siliconrt_decode_state_t* state,
    uint32_t logical_token_count,
    uint32_t resident_token_count,
    uint64_t sequence_bytes) {
  return set_decode_state_residency_impl(
      state,
      logical_token_count,
      resident_token_count,
      sequence_bytes,
      true);
}

void siliconrt_decode_state_destroy(
    siliconrt_arena_t*,
    siliconrt_budget_t*,
    siliconrt_decode_state_t* state) {
  if (state == nullptr) {
    return;
  }

  const auto allocation = owned_allocation_from_state(state);
  release_decode_owned_allocation(
      state->owner_arena, state->owner_budget, allocation);
  if (state->borrowed_prefix != nullptr &&
      state->borrowed_prefix->active_borrowed_decode_count > 0) {
    state->borrowed_prefix->active_borrowed_decode_count -= 1;
    maybe_finalize_prefix_handle(state->borrowed_prefix);
  }
  delete state;
}

siliconrt_status_t siliconrt_prefill_begin(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const char* model_key,
    siliconrt_prefill_handle_t** out_prefill) {
  if (arena == nullptr || budget == nullptr || model_key == nullptr ||
      out_prefill == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_prefill = nullptr;

  try {
    auto* prefill = new siliconrt_prefill_handle_t();
    prefill->arena = arena;
    prefill->budget = budget;
    prefill->model_key = model_key;
    *out_prefill = prefill;
    return SILICONRT_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return SILICONRT_STATUS_OUT_OF_MEMORY;
  }
}

void siliconrt_prefill_destroy(
    siliconrt_arena_t*,
    siliconrt_budget_t*,
    siliconrt_prefill_handle_t* prefill) {
  delete prefill;
}

siliconrt_status_t siliconrt_prefill_finish_as_prefix(
    siliconrt_prefill_handle_t* prefill,
    const siliconrt_prefix_descriptor_t* descriptor,
    siliconrt_prefix_handle_t** out_handle) {
  if (prefill == nullptr || out_handle == nullptr) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  *out_handle = nullptr;
  if (prefill->finished) {
    return SILICONRT_STATUS_INVALID_ARGUMENT;
  }
  const auto descriptor_status = validate_descriptor(descriptor);
  if (descriptor_status != SILICONRT_STATUS_OK) {
    return descriptor_status;
  }
  if (prefill->model_key != descriptor->model_key) {
    return SILICONRT_STATUS_INCOMPATIBLE;
  }

  const auto status = siliconrt_prefix_create(
      prefill->arena,
      prefill->budget,
      descriptor,
      out_handle);
  if (status == SILICONRT_STATUS_OK) {
    prefill->finished = true;
  }
  return status;
}

}  // extern "C"
