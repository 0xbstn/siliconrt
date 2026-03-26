#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "siliconrt/circular_sequence_plan.hpp"
#include "siliconrt/metal_backing_store.hpp"

namespace siliconrt {

struct MetalWindowSegment {
  MetalBufferSlice source_slice = {};
  void* metal_buffer = nullptr;
  std::size_t offset_bytes = 0;
  std::span<std::uint8_t> bytes = {};

  [[nodiscard]] bool present() const {
    return metal_buffer != nullptr && !bytes.empty();
  }

  [[nodiscard]] std::size_t size_bytes() const {
    return bytes.size();
  }
};

struct MetalWindowSegmentPair {
  MetalWindowSegment first = {};
  MetalWindowSegment second = {};

  [[nodiscard]] std::size_t total_bytes() const {
    return first.size_bytes() + second.size_bytes();
  }

  [[nodiscard]] std::size_t segment_count() const {
    return (first.present() ? 1 : 0) + (second.present() ? 1 : 0);
  }
};

[[nodiscard]] inline MetalWindowSegmentPair make_metal_window_segments(
    const MetalBufferSlice& slice,
    const CircularSequenceState& state) {
  if (!slice.valid() || !state.valid() || state.capacity_bytes > slice.writable_bytes.size()) {
    return {};
  }

  const auto visible = make_circular_visible_segments(state);
  MetalWindowSegmentPair segments;
  if (visible.first.present()) {
    segments.first = MetalWindowSegment{
        .source_slice = slice,
        .metal_buffer = slice.metal_buffer,
        .offset_bytes = visible.first.offset_bytes,
        .bytes = slice.writable_bytes.subspan(
            visible.first.offset_bytes, visible.first.size_bytes),
    };
  }
  if (visible.second.present()) {
    segments.second = MetalWindowSegment{
        .source_slice = slice,
        .metal_buffer = slice.metal_buffer,
        .offset_bytes = visible.second.offset_bytes,
        .bytes = slice.writable_bytes.subspan(
            visible.second.offset_bytes, visible.second.size_bytes),
    };
  }
  return segments;
}

}  // namespace siliconrt
