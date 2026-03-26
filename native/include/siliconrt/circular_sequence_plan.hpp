#pragma once

#include <algorithm>
#include <cstddef>

namespace siliconrt {

struct CircularSegment {
  std::size_t offset_bytes = 0;
  std::size_t size_bytes = 0;

  [[nodiscard]] constexpr bool present() const {
    return size_bytes != 0;
  }
};

struct CircularSegmentPair {
  CircularSegment first = {};
  CircularSegment second = {};

  [[nodiscard]] constexpr std::size_t total_bytes() const {
    return first.size_bytes + second.size_bytes;
  }

  [[nodiscard]] constexpr std::size_t segment_count() const {
    return (first.present() ? 1 : 0) + (second.present() ? 1 : 0);
  }
};

struct CircularSequenceState {
  std::size_t head_offset_bytes = 0;
  std::size_t used_bytes = 0;
  std::size_t capacity_bytes = 0;

  [[nodiscard]] constexpr bool valid() const {
    if (capacity_bytes == 0) {
      return head_offset_bytes == 0 && used_bytes == 0;
    }
    return head_offset_bytes < capacity_bytes && used_bytes <= capacity_bytes;
  }
};

struct CircularAppendPlan {
  CircularSequenceState before = {};
  CircularSequenceState after = {};
  std::size_t append_source_offset_bytes = 0;
  std::size_t append_bytes = 0;
  std::size_t dropped_prefix_bytes = 0;
  CircularSegmentPair append_segments = {};

  [[nodiscard]] constexpr bool wraps() const {
    return append_segments.second.present();
  }
};

[[nodiscard]] constexpr CircularSegmentPair make_circular_visible_segments(
    const CircularSequenceState& state) {
  if (!state.valid() || state.used_bytes == 0 || state.capacity_bytes == 0) {
    return {};
  }

  const auto first_size =
      std::min(state.used_bytes, state.capacity_bytes - state.head_offset_bytes);
  return CircularSegmentPair{
      .first =
          CircularSegment{
              .offset_bytes = state.head_offset_bytes,
              .size_bytes = first_size,
          },
      .second =
          CircularSegment{
              .offset_bytes = 0,
              .size_bytes = state.used_bytes - first_size,
          },
  };
}

[[nodiscard]] constexpr CircularAppendPlan make_circular_append_plan(
    const CircularSequenceState& before,
    std::size_t append_bytes) {
  CircularAppendPlan plan;
  plan.before = before;
  if (!before.valid() || before.capacity_bytes == 0) {
    return plan;
  }

  const auto bounded_append_bytes =
      append_bytes >= before.capacity_bytes ? before.capacity_bytes : append_bytes;
  plan.append_bytes = bounded_append_bytes;
  plan.append_source_offset_bytes =
      append_bytes > before.capacity_bytes ? append_bytes - before.capacity_bytes : 0;

  if (bounded_append_bytes == 0) {
    plan.after = before;
    return plan;
  }

  if (bounded_append_bytes == before.capacity_bytes) {
    plan.dropped_prefix_bytes = before.used_bytes;
    plan.append_segments.first = CircularSegment{
        .offset_bytes = 0,
        .size_bytes = before.capacity_bytes,
    };
    plan.after = CircularSequenceState{
        .head_offset_bytes = 0,
        .used_bytes = before.capacity_bytes,
        .capacity_bytes = before.capacity_bytes,
    };
    return plan;
  }

  const auto append_begin =
      (before.head_offset_bytes + before.used_bytes) % before.capacity_bytes;
  const auto first_append_size =
      std::min(bounded_append_bytes, before.capacity_bytes - append_begin);
  plan.append_segments.first = CircularSegment{
      .offset_bytes = append_begin,
      .size_bytes = first_append_size,
  };
  plan.append_segments.second = CircularSegment{
      .offset_bytes = 0,
      .size_bytes = bounded_append_bytes - first_append_size,
  };

  const auto total_bytes = before.used_bytes + bounded_append_bytes;
  const auto dropped_prefix_bytes =
      total_bytes > before.capacity_bytes ? total_bytes - before.capacity_bytes : 0;
  plan.dropped_prefix_bytes = dropped_prefix_bytes;
  plan.after = CircularSequenceState{
      .head_offset_bytes =
          (before.head_offset_bytes + dropped_prefix_bytes) % before.capacity_bytes,
      .used_bytes = std::min(before.capacity_bytes, total_bytes),
      .capacity_bytes = before.capacity_bytes,
  };
  return plan;
}

}  // namespace siliconrt
