#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "siliconrt/kv_arena.hpp"

namespace siliconrt {

struct PartitionedKvArenaStats {
  std::uint64_t total_capacity_bytes = 0;
  std::uint64_t total_free_bytes = 0;
  std::uint64_t total_used_bytes = 0;
  std::uint64_t total_allocated_capacity_bytes = 0;
  std::uint64_t total_allocated_span_count = 0;

  KvArenaStats sequence_pool = {};
  KvArenaStats constant_pool = {};
};

struct PartitionedSpanView {
  KvSpan global_span = {};
  std::uint64_t backing_offset_bytes = 0;
};

class PartitionedKvArena {
 public:
  PartitionedKvArena(
      std::uint64_t sequence_capacity_bytes,
      std::uint64_t constant_capacity_bytes)
      : sequence_capacity_bytes_(sequence_capacity_bytes),
        constant_capacity_bytes_(constant_capacity_bytes),
        sequence_arena_(sequence_capacity_bytes),
        constant_arena_(constant_capacity_bytes) {}

  [[nodiscard]] std::uint64_t total_capacity_bytes() const {
    return sequence_capacity_bytes_ + constant_capacity_bytes_;
  }

  [[nodiscard]] std::uint64_t sequence_capacity_bytes() const {
    return sequence_capacity_bytes_;
  }

  [[nodiscard]] std::uint64_t constant_capacity_bytes() const {
    return constant_capacity_bytes_;
  }

  std::optional<KvSpan> allocate(
      std::uint64_t capacity_bytes,
      std::uint32_t token_capacity,
      ResidencyClass residency_class) {
    if (residency_class == ResidencyClass::kSequenceGrowing) {
      return allocate_from_pool(
          sequence_arena_,
          PoolKind::kSequence,
          0,
          capacity_bytes,
          token_capacity,
          residency_class);
    }
    if (residency_class == ResidencyClass::kConstantState) {
      return allocate_from_pool(
          constant_arena_,
          PoolKind::kConstant,
          sequence_capacity_bytes_,
          capacity_bytes,
          token_capacity,
          residency_class);
    }
    return std::nullopt;
  }

  bool commit(
      std::uint64_t span_id,
      std::uint64_t used_bytes,
      std::uint32_t token_count) {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return false;
    }
    return pool_arena(it->second.kind)->commit(
        it->second.local_span_id, used_bytes, token_count);
  }

  bool release(std::uint64_t span_id) {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return false;
    }
    const auto ok = pool_arena(it->second.kind)->release(it->second.local_span_id);
    spans_.erase(it);
    return ok;
  }

  [[nodiscard]] std::optional<KvSpan> get(std::uint64_t span_id) const {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return std::nullopt;
    }
    const auto local = const_pool_arena(it->second.kind)->get(it->second.local_span_id);
    if (!local.has_value()) {
      return std::nullopt;
    }
    KvSpan out = *local;
    out.span_id = span_id;
    out.offset_bytes += it->second.base_offset_bytes;
    return out;
  }

  [[nodiscard]] std::optional<PartitionedSpanView> describe(
      std::uint64_t span_id) const {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return std::nullopt;
    }
    const auto local = const_pool_arena(it->second.kind)->get(it->second.local_span_id);
    if (!local.has_value()) {
      return std::nullopt;
    }

    PartitionedSpanView out;
    out.global_span = *local;
    out.global_span.span_id = span_id;
    out.global_span.offset_bytes += it->second.base_offset_bytes;
    out.backing_offset_bytes = local->offset_bytes;
    return out;
  }

  [[nodiscard]] PartitionedKvArenaStats stats() const {
    PartitionedKvArenaStats out;
    out.sequence_pool = sequence_arena_.stats();
    out.constant_pool = constant_arena_.stats();
    out.total_capacity_bytes = total_capacity_bytes();
    out.total_free_bytes = out.sequence_pool.free_bytes + out.constant_pool.free_bytes;
    out.total_used_bytes = out.sequence_pool.used_bytes + out.constant_pool.used_bytes;
    out.total_allocated_capacity_bytes =
        out.sequence_pool.allocated_capacity_bytes +
        out.constant_pool.allocated_capacity_bytes;
    out.total_allocated_span_count =
        out.sequence_pool.allocated_span_count +
        out.constant_pool.allocated_span_count;
    return out;
  }

 private:
  enum class PoolKind : std::uint8_t {
    kSequence = 1,
    kConstant = 2,
  };

  struct SpanLocation {
    PoolKind kind;
    std::uint64_t local_span_id = 0;
    std::uint64_t base_offset_bytes = 0;
  };

  std::optional<KvSpan> allocate_from_pool(
      KvArena& arena,
      PoolKind kind,
      std::uint64_t base_offset_bytes,
      std::uint64_t capacity_bytes,
      std::uint32_t token_capacity,
      ResidencyClass residency_class) {
    auto local = arena.allocate(capacity_bytes, token_capacity, residency_class);
    if (!local.has_value()) {
      return std::nullopt;
    }

    const auto global_span_id = next_span_id_++;
    spans_.emplace(
        global_span_id,
        SpanLocation{
            .kind = kind,
            .local_span_id = local->span_id,
            .base_offset_bytes = base_offset_bytes,
        });

    KvSpan out = *local;
    out.span_id = global_span_id;
    out.offset_bytes += base_offset_bytes;
    return out;
  }

  KvArena* pool_arena(PoolKind kind) {
    return kind == PoolKind::kSequence ? &sequence_arena_ : &constant_arena_;
  }

  const KvArena* const_pool_arena(PoolKind kind) const {
    return kind == PoolKind::kSequence ? &sequence_arena_ : &constant_arena_;
  }

  std::uint64_t sequence_capacity_bytes_ = 0;
  std::uint64_t constant_capacity_bytes_ = 0;
  std::uint64_t next_span_id_ = 1;
  KvArena sequence_arena_;
  KvArena constant_arena_;
  std::unordered_map<std::uint64_t, SpanLocation> spans_;
};

}  // namespace siliconrt
