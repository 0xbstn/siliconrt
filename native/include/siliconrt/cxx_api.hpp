#pragma once

#include <memory>
#include <stdexcept>

#include "siliconrt/c_api.h"
#include "siliconrt/storage_layout.hpp"

namespace siliconrt {

namespace detail {

struct BudgetDeleter {
  void operator()(siliconrt_budget_t* budget) const {
    siliconrt_budget_destroy(budget);
  }
};

struct ArenaDeleter {
  void operator()(siliconrt_arena_t* arena) const {
    siliconrt_arena_destroy(arena);
  }
};

struct PrefixHandleDeleter {
  siliconrt_arena_t* arena = nullptr;
  siliconrt_budget_t* budget = nullptr;

  void operator()(siliconrt_prefix_handle_t* handle) const {
    siliconrt_prefix_destroy(arena, budget, handle);
  }
};

struct DecodeStateDeleter {
  siliconrt_arena_t* arena = nullptr;
  siliconrt_budget_t* budget = nullptr;

  void operator()(siliconrt_decode_state_t* state) const {
    siliconrt_decode_state_destroy(arena, budget, state);
  }
};

}  // namespace detail

using UniqueBudget = std::unique_ptr<siliconrt_budget_t, detail::BudgetDeleter>;
using UniqueArena = std::unique_ptr<siliconrt_arena_t, detail::ArenaDeleter>;
using UniquePrefixHandle =
    std::unique_ptr<siliconrt_prefix_handle_t, detail::PrefixHandleDeleter>;
using UniqueDecodeState =
    std::unique_ptr<siliconrt_decode_state_t, detail::DecodeStateDeleter>;

inline UniqueBudget make_budget(std::uint64_t capacity_bytes) {
  siliconrt_budget_t* raw = nullptr;
  const auto status = siliconrt_budget_create(capacity_bytes, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_budget_create failed");
  }
  return UniqueBudget(raw);
}

inline UniqueArena make_arena(std::uint64_t capacity_bytes) {
  siliconrt_arena_t* raw = nullptr;
  const auto status = siliconrt_arena_create(capacity_bytes, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_arena_create failed");
  }
  return UniqueArena(raw);
}

inline UniqueArena make_partitioned_arena(
    std::uint64_t sequence_capacity_bytes,
    std::uint64_t constant_capacity_bytes) {
  siliconrt_arena_t* raw = nullptr;
  const auto status = siliconrt_arena_create_partitioned(
      sequence_capacity_bytes, constant_capacity_bytes, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_arena_create_partitioned failed");
  }
  return UniqueArena(raw);
}

inline UniquePrefixHandle make_prefix_handle(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_descriptor_t& descriptor) {
  siliconrt_prefix_handle_t* raw = nullptr;
  const auto status = siliconrt_prefix_create(arena, budget, &descriptor, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_prefix_create failed");
  }
  return UniquePrefixHandle(raw, detail::PrefixHandleDeleter{arena, budget});
}

inline UniqueDecodeState make_decode_state(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle) {
  siliconrt_decode_state_t* raw = nullptr;
  const auto status = siliconrt_decode_restore(arena, budget, handle, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_decode_restore failed");
  }
  return UniqueDecodeState(raw, detail::DecodeStateDeleter{arena, budget});
}

inline UniqueDecodeState make_borrowed_decode_state(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    const siliconrt_prefix_handle_t* handle) {
  siliconrt_decode_state_t* raw = nullptr;
  const auto status =
      siliconrt_decode_restore_borrowed(arena, budget, handle, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_decode_restore_borrowed failed");
  }
  return UniqueDecodeState(raw, detail::DecodeStateDeleter{arena, budget});
}

inline void promote_decode_sequence(siliconrt_decode_state_t* state) {
  const auto status = siliconrt_decode_state_promote_sequence(state);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_decode_state_promote_sequence failed");
  }
}

inline void set_decode_state_residency_promoting(
    siliconrt_decode_state_t* state,
    std::uint32_t logical_token_count,
    std::uint32_t resident_token_count,
    std::uint64_t sequence_bytes) {
  const auto status = siliconrt_decode_state_set_residency_promoting(
      state, logical_token_count, resident_token_count, sequence_bytes);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error(
        "siliconrt_decode_state_set_residency_promoting failed");
  }
}

inline StorageLayoutView describe_prefix_storage(
    const siliconrt_prefix_handle_t* handle) {
  siliconrt_storage_layout_t raw = {};
  const auto status = siliconrt_prefix_describe_storage(handle, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_prefix_describe_storage failed");
  }
  return StorageLayoutView{raw};
}

inline StorageLayoutView describe_decode_storage(
    const siliconrt_decode_state_t* state) {
  siliconrt_storage_layout_t raw = {};
  const auto status = siliconrt_decode_state_describe_storage(state, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_decode_state_describe_storage failed");
  }
  return StorageLayoutView{raw};
}

inline BackingStoreLayoutView describe_arena_backing_stores(
    const siliconrt_arena_t* arena) {
  siliconrt_backing_store_layout_t raw = {};
  const auto status = siliconrt_arena_describe_backing_stores(arena, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_arena_describe_backing_stores failed");
  }
  return BackingStoreLayoutView{raw};
}

inline StorageHandleLayoutView describe_prefix_storage_handles(
    const siliconrt_prefix_handle_t* handle) {
  siliconrt_storage_handle_layout_t raw = {};
  const auto status = siliconrt_prefix_describe_storage_handles(handle, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error("siliconrt_prefix_describe_storage_handles failed");
  }
  return StorageHandleLayoutView{raw};
}

inline StorageHandleLayoutView describe_decode_storage_handles(
    const siliconrt_decode_state_t* state) {
  siliconrt_storage_handle_layout_t raw = {};
  const auto status =
      siliconrt_decode_state_describe_storage_handles(state, &raw);
  if (status != SILICONRT_STATUS_OK) {
    throw std::runtime_error(
        "siliconrt_decode_state_describe_storage_handles failed");
  }
  return StorageHandleLayoutView{raw};
}

}  // namespace siliconrt
