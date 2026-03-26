#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "siliconrt/kv_types.hpp"

namespace siliconrt {

struct KvArenaStats {
  std::uint64_t capacity_bytes = 0;
  std::uint64_t free_bytes = 0;
  std::uint64_t largest_free_range_bytes = 0;
  std::uint64_t allocated_capacity_bytes = 0;
  std::uint64_t used_bytes = 0;
  std::uint64_t allocated_span_count = 0;
};

class KvArena {
 public:
  explicit KvArena(std::uint64_t capacity_bytes)
      : capacity_bytes_(capacity_bytes),
        free_ranges_({FreeRange{0, capacity_bytes}}) {}

  [[nodiscard]] std::uint64_t capacity_bytes() const { return capacity_bytes_; }

  std::optional<KvSpan> allocate(
      std::uint64_t capacity_bytes,
      std::uint32_t token_capacity,
      ResidencyClass residency_class) {
    for (std::size_t i = 0; i < free_ranges_.size(); ++i) {
      auto& range = free_ranges_[i];
      if (range.length_bytes < capacity_bytes) {
        continue;
      }

      KvSpan span;
      span.span_id = next_span_id_++;
      span.offset_bytes = range.offset_bytes;
      span.capacity_bytes = capacity_bytes;
      span.used_bytes = 0;
      span.token_capacity = token_capacity;
      span.token_count = 0;
      span.residency_class = residency_class;
      span.in_use = true;

      range.offset_bytes += capacity_bytes;
      range.length_bytes -= capacity_bytes;
      if (range.length_bytes == 0) {
        free_ranges_.erase(free_ranges_.begin() + static_cast<long>(i));
      }

      spans_.emplace(span.span_id, span);
      return span;
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
    auto& span = it->second;
    if (used_bytes > span.capacity_bytes || token_count > span.token_capacity) {
      return false;
    }
    span.used_bytes = used_bytes;
    span.token_count = token_count;
    return true;
  }

  bool release(std::uint64_t span_id) {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return false;
    }
    const auto span = it->second;
    spans_.erase(it);
    free_ranges_.push_back({span.offset_bytes, span.capacity_bytes});
    coalesce_free_ranges();
    return true;
  }

  [[nodiscard]] bool contains(std::uint64_t span_id) const {
    return spans_.find(span_id) != spans_.end();
  }

  [[nodiscard]] std::optional<KvSpan> get(std::uint64_t span_id) const {
    auto it = spans_.find(span_id);
    if (it == spans_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  [[nodiscard]] KvArenaStats stats() const {
    KvArenaStats out;
    out.capacity_bytes = capacity_bytes_;
    out.allocated_span_count = spans_.size();

    for (const auto& range : free_ranges_) {
      out.free_bytes += range.length_bytes;
      out.largest_free_range_bytes =
          std::max(out.largest_free_range_bytes, range.length_bytes);
    }

    for (const auto& [_, span] : spans_) {
      out.allocated_capacity_bytes += span.capacity_bytes;
      out.used_bytes += span.used_bytes;
    }

    return out;
  }

 private:
  struct FreeRange {
    std::uint64_t offset_bytes = 0;
    std::uint64_t length_bytes = 0;
  };

  void coalesce_free_ranges() {
    std::sort(
        free_ranges_.begin(),
        free_ranges_.end(),
        [](const FreeRange& lhs, const FreeRange& rhs) {
          return lhs.offset_bytes < rhs.offset_bytes;
        });

    std::vector<FreeRange> merged;
    merged.reserve(free_ranges_.size());
    for (const auto& range : free_ranges_) {
      if (merged.empty()) {
        merged.push_back(range);
        continue;
      }
      auto& last = merged.back();
      if (last.offset_bytes + last.length_bytes == range.offset_bytes) {
        last.length_bytes += range.length_bytes;
      } else {
        merged.push_back(range);
      }
    }
    free_ranges_ = std::move(merged);
  }

  std::uint64_t capacity_bytes_ = 0;
  std::uint64_t next_span_id_ = 1;
  std::vector<FreeRange> free_ranges_;
  std::unordered_map<std::uint64_t, KvSpan> spans_;
};

}  // namespace siliconrt
