#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "siliconrt/kv_budget.hpp"
#include "siliconrt/kv_types.hpp"
#include "siliconrt/partitioned_kv_arena.hpp"
#include "siliconrt/partitioned_runtime_plan.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"
#include "siliconrt/restore_plan.hpp"

namespace siliconrt {

struct AliasedDecodeHandle {
  std::uint64_t handle_id = 0;
  std::uint64_t parent_prefix_handle_id = 0;
  std::uint64_t sequence_span_id = 0;
  std::uint64_t borrowed_sequence_span_id = 0;
  std::uint64_t borrowed_constant_span_id = 0;
  std::uint32_t logical_token_count = 0;
  std::uint32_t resident_token_count = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t borrowed_sequence_bytes = 0;
  std::uint64_t borrowed_constant_bytes = 0;
  RestoreAliasMode mode = RestoreAliasMode::kShareConstantState;

  [[nodiscard]] constexpr bool owns_sequence() const {
    return sequence_span_id != 0;
  }

  [[nodiscard]] constexpr bool borrows_sequence() const {
    return borrowed_sequence_span_id != 0;
  }
};

struct PartitionedPrefixStoreStats {
  std::uint64_t handle_count = 0;
  std::uint64_t decode_handle_count = 0;
  std::uint64_t total_capacity_bytes = 0;
  std::uint64_t committed_bytes = 0;
  std::uint64_t available_bytes = 0;
  std::uint64_t sequence_capacity_bytes = 0;
  std::uint64_t constant_capacity_bytes = 0;
  PartitionedKvArenaStats arena = {};
};

class PartitionedPrefixStore {
 public:
  explicit PartitionedPrefixStore(PartitionedRuntimePlan plan)
      : plan_(plan),
        budget_(plan.total_capacity_bytes),
        arena_(plan.sequence_capacity_bytes, plan.constant_capacity_bytes) {}

  [[nodiscard]] const PartitionedRuntimePlan& plan() const { return plan_; }

  [[nodiscard]] std::optional<PrefixHandle> materialize(
      const OwnedPrefixDescriptor& descriptor) {
    if (descriptor.cache_mode != SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS) {
      return std::nullopt;
    }
    if (descriptor.resident_token_count > descriptor.logical_token_count) {
      return std::nullopt;
    }
    const auto total_bytes = descriptor.total_bytes();
    if (!budget_.reserve(total_bytes)) {
      return std::nullopt;
    }

    AllocationRecord allocation;
    allocation.sequence_bytes = descriptor.sequence_bytes;
    allocation.constant_bytes = descriptor.constant_bytes;

    if (descriptor.sequence_bytes > 0) {
      auto sequence_span = arena_.allocate(
          descriptor.sequence_bytes,
          descriptor.resident_token_count,
          ResidencyClass::kSequenceGrowing);
      if (!sequence_span.has_value()) {
        budget_.release_reserved(total_bytes);
        return std::nullopt;
      }
      if (!arena_.commit(
              sequence_span->span_id,
              descriptor.sequence_bytes,
              descriptor.resident_token_count)) {
        arena_.release(sequence_span->span_id);
        budget_.release_reserved(total_bytes);
        return std::nullopt;
      }
      allocation.sequence_span_id = sequence_span->span_id;
    }

    if (descriptor.constant_bytes > 0) {
      auto constant_span = arena_.allocate(
          descriptor.constant_bytes,
          0,
          ResidencyClass::kConstantState);
      if (!constant_span.has_value()) {
        rollback(allocation);
        budget_.release_reserved(total_bytes);
        return std::nullopt;
      }
      if (!arena_.commit(
              constant_span->span_id, descriptor.constant_bytes, 0)) {
        arena_.release(constant_span->span_id);
        rollback(allocation);
        budget_.release_reserved(total_bytes);
        return std::nullopt;
      }
      allocation.constant_span_id = constant_span->span_id;
    }

    if (!budget_.commit_reserved(total_bytes)) {
      rollback(allocation);
      budget_.release_reserved(total_bytes);
      return std::nullopt;
    }

    PrefixHandle handle;
    handle.handle_id = next_handle_id_++;
    handle.model_key = descriptor.model_key;
    handle.prefix_hash_hex = descriptor.prefix_hash_hex;
    handle.cache_mode = CacheMode::kBoundedContiguous;
    handle.sequence_span_id = allocation.sequence_span_id;
    handle.constant_span_id = allocation.constant_span_id;
    handle.logical_token_count = descriptor.logical_token_count;
    handle.resident_token_count = descriptor.resident_token_count;
    handle.sequence_bytes = descriptor.sequence_bytes;
    handle.constant_bytes = descriptor.constant_bytes;

    handles_.emplace(handle.handle_id, handle);
    return handle;
  }

  [[nodiscard]] std::optional<PrefixHandle> get(std::uint64_t handle_id) const {
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  [[nodiscard]] std::optional<AliasedDecodeHandle> restore_share_constant(
      std::uint64_t handle_id) {
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return std::nullopt;
    }

    const auto plan = RestorePlanner::make_plan(
        to_owned_descriptor(it->second), RestoreAliasMode::kShareConstantState);
    if (!budget_.reserve(plan.additional_bytes())) {
      return std::nullopt;
    }

    auto sequence_span = arena_.allocate(
        plan.clone_sequence_bytes,
        plan.resident_token_count,
        ResidencyClass::kSequenceGrowing);
    if (!sequence_span.has_value()) {
      budget_.release_reserved(plan.additional_bytes());
      return std::nullopt;
    }
    if (!arena_.commit(
            sequence_span->span_id,
            plan.clone_sequence_bytes,
            plan.resident_token_count)) {
      arena_.release(sequence_span->span_id);
      budget_.release_reserved(plan.additional_bytes());
      return std::nullopt;
    }
    if (!budget_.commit_reserved(plan.additional_bytes())) {
      arena_.release(sequence_span->span_id);
      budget_.release_reserved(plan.additional_bytes());
      return std::nullopt;
    }

    AliasedDecodeHandle decode;
    decode.handle_id = next_decode_handle_id_++;
    decode.parent_prefix_handle_id = handle_id;
    decode.sequence_span_id = sequence_span->span_id;
    decode.borrowed_constant_span_id = it->second.constant_span_id;
    decode.logical_token_count = it->second.logical_token_count;
    decode.resident_token_count = it->second.resident_token_count;
    decode.sequence_bytes = plan.clone_sequence_bytes;
    decode.borrowed_constant_bytes = plan.borrowed_constant_bytes;
    decode.mode = plan.mode;

    decode_handles_.emplace(decode.handle_id, decode);
    active_decode_counts_[handle_id] += 1;
    return decode;
  }

  [[nodiscard]] std::optional<AliasedDecodeHandle> restore_borrow_until_append(
      std::uint64_t handle_id) {
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return std::nullopt;
    }

    const auto plan = RestorePlanner::make_plan(
        to_owned_descriptor(it->second),
        RestoreAliasMode::kBorrowSequenceAndConstant);

    AliasedDecodeHandle decode;
    decode.handle_id = next_decode_handle_id_++;
    decode.parent_prefix_handle_id = handle_id;
    decode.borrowed_sequence_span_id = it->second.sequence_span_id;
    decode.borrowed_constant_span_id = it->second.constant_span_id;
    decode.logical_token_count = it->second.logical_token_count;
    decode.resident_token_count = it->second.resident_token_count;
    decode.borrowed_sequence_bytes = plan.borrowed_sequence_bytes;
    decode.borrowed_constant_bytes = plan.borrowed_constant_bytes;
    decode.mode = plan.mode;

    decode_handles_.emplace(decode.handle_id, decode);
    active_decode_counts_[handle_id] += 1;
    return decode;
  }

  [[nodiscard]] std::optional<AliasedDecodeHandle> get_decode(
      std::uint64_t handle_id) const {
    auto it = decode_handles_.find(handle_id);
    if (it == decode_handles_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  bool release_decode(std::uint64_t handle_id) {
    auto it = decode_handles_.find(handle_id);
    if (it == decode_handles_.end()) {
      return false;
    }

    if (it->second.sequence_span_id != 0) {
      arena_.release(it->second.sequence_span_id);
    }
    budget_.release_committed(it->second.sequence_bytes);
    auto count_it = active_decode_counts_.find(it->second.parent_prefix_handle_id);
    if (count_it != active_decode_counts_.end()) {
      if (count_it->second > 0) {
        count_it->second -= 1;
      }
      if (count_it->second == 0) {
        active_decode_counts_.erase(count_it);
      }
    }
    decode_handles_.erase(it);
    return true;
  }

  bool promote_decode_sequence(std::uint64_t handle_id) {
    auto it = decode_handles_.find(handle_id);
    if (it == decode_handles_.end()) {
      return false;
    }
    auto& decode = it->second;
    if (!decode.borrows_sequence() || decode.owns_sequence()) {
      return decode.owns_sequence();
    }
    if (!budget_.reserve(decode.borrowed_sequence_bytes)) {
      return false;
    }
    auto sequence_span = arena_.allocate(
        decode.borrowed_sequence_bytes,
        decode.resident_token_count,
        ResidencyClass::kSequenceGrowing);
    if (!sequence_span.has_value()) {
      budget_.release_reserved(decode.borrowed_sequence_bytes);
      return false;
    }
    if (!arena_.commit(
            sequence_span->span_id,
            decode.borrowed_sequence_bytes,
            decode.resident_token_count)) {
      arena_.release(sequence_span->span_id);
      budget_.release_reserved(decode.borrowed_sequence_bytes);
      return false;
    }
    if (!budget_.commit_reserved(decode.borrowed_sequence_bytes)) {
      arena_.release(sequence_span->span_id);
      budget_.release_reserved(decode.borrowed_sequence_bytes);
      return false;
    }

    decode.sequence_span_id = sequence_span->span_id;
    decode.sequence_bytes = decode.borrowed_sequence_bytes;
    return true;
  }

  [[nodiscard]] std::uint64_t active_decode_count(std::uint64_t handle_id) const {
    auto it = active_decode_counts_.find(handle_id);
    return it == active_decode_counts_.end() ? 0 : it->second;
  }

  bool release(std::uint64_t handle_id) {
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return false;
    }
    if (active_decode_count(handle_id) != 0) {
      return false;
    }

    if (it->second.sequence_span_id != 0) {
      arena_.release(it->second.sequence_span_id);
    }
    if (it->second.constant_span_id != 0) {
      arena_.release(it->second.constant_span_id);
    }
    budget_.release_committed(it->second.total_bytes());
    handles_.erase(it);
    return true;
  }

  [[nodiscard]] PartitionedPrefixStoreStats stats() const {
    const auto arena_stats = arena_.stats();
    return PartitionedPrefixStoreStats{
        .handle_count = handles_.size(),
        .decode_handle_count = decode_handles_.size(),
        .total_capacity_bytes = budget_.capacity_bytes(),
        .committed_bytes = budget_.committed_bytes(),
        .available_bytes = budget_.available_bytes(),
        .sequence_capacity_bytes = plan_.sequence_capacity_bytes,
        .constant_capacity_bytes = plan_.constant_capacity_bytes,
        .arena = arena_stats,
    };
  }

 private:
  struct AllocationRecord {
    std::uint64_t sequence_span_id = 0;
    std::uint64_t constant_span_id = 0;
    std::uint64_t sequence_bytes = 0;
    std::uint64_t constant_bytes = 0;
  };

  void rollback(const AllocationRecord& allocation) {
    if (allocation.sequence_span_id != 0) {
      arena_.release(allocation.sequence_span_id);
    }
    if (allocation.constant_span_id != 0) {
      arena_.release(allocation.constant_span_id);
    }
  }

  [[nodiscard]] static OwnedPrefixDescriptor to_owned_descriptor(
      const PrefixHandle& handle) {
    return OwnedPrefixDescriptor{
        .model_key = handle.model_key,
        .prefix_hash_hex = handle.prefix_hash_hex,
        .logical_token_count = handle.logical_token_count,
        .resident_token_count = handle.resident_token_count,
        .sequence_bytes = handle.sequence_bytes,
        .constant_bytes = handle.constant_bytes,
        .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
    };
  }

  static inline std::atomic<std::uint64_t> next_handle_id_{1};
  static inline std::atomic<std::uint64_t> next_decode_handle_id_{1};

  PartitionedRuntimePlan plan_;
  KvBudget budget_;
  PartitionedKvArena arena_;
  std::unordered_map<std::uint64_t, PrefixHandle> handles_;
  std::unordered_map<std::uint64_t, AliasedDecodeHandle> decode_handles_;
  std::unordered_map<std::uint64_t, std::uint64_t> active_decode_counts_;
};

class PartitionedBoundedRuntime {
 public:
  PartitionedBoundedRuntime(
      PrefixDescriptorBuilder builder,
      PartitionedRuntimePlan plan)
      : builder_(std::move(builder)), store_(plan) {}

  [[nodiscard]] const PrefixDescriptorBuilder& builder() const {
    return builder_;
  }

  [[nodiscard]] const PartitionedPrefixStore& store() const { return store_; }
  [[nodiscard]] PartitionedPrefixStore& store() { return store_; }

  [[nodiscard]] OwnedPrefixDescriptor make_prefix_descriptor(
      std::string prefix_hash_hex,
      std::uint32_t logical_token_count) const {
    return builder_.make_prefix(std::move(prefix_hash_hex), logical_token_count);
  }

  [[nodiscard]] std::optional<PrefixHandle> materialize_prefix(
      const OwnedPrefixDescriptor& descriptor) {
    return store_.materialize(descriptor);
  }

  bool release_prefix(std::uint64_t handle_id) {
    return store_.release(handle_id);
  }

  [[nodiscard]] std::optional<AliasedDecodeHandle> restore_share_constant(
      std::uint64_t handle_id) {
    return store_.restore_share_constant(handle_id);
  }

  [[nodiscard]] std::optional<AliasedDecodeHandle> restore_borrow_until_append(
      std::uint64_t handle_id) {
    return store_.restore_borrow_until_append(handle_id);
  }

  bool promote_decode_sequence(std::uint64_t handle_id) {
    return store_.promote_decode_sequence(handle_id);
  }

  bool release_decode(std::uint64_t handle_id) {
    return store_.release_decode(handle_id);
  }

 private:
  PrefixDescriptorBuilder builder_;
  PartitionedPrefixStore store_;
};

}  // namespace siliconrt
