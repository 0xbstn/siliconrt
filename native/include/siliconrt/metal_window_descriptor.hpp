#pragma once

#include <cstddef>
#include <cstdint>

#include "siliconrt/metal_window_segments.hpp"

namespace siliconrt {

struct MetalWindowRangeDescriptor {
  std::size_t offset_bytes = 0;
  std::size_t size_bytes = 0;

  [[nodiscard]] constexpr bool present() const {
    return size_bytes != 0;
  }
};

struct MetalWindowDescriptor {
  void* metal_buffer = nullptr;
  std::uint32_t segment_count = 0;
  MetalWindowRangeDescriptor first = {};
  MetalWindowRangeDescriptor second = {};

  [[nodiscard]] constexpr std::size_t total_bytes() const {
    return first.size_bytes + second.size_bytes;
  }

  [[nodiscard]] constexpr bool linear() const {
    return segment_count <= 1;
  }

  [[nodiscard]] constexpr bool valid() const {
    return (segment_count == 0 && metal_buffer == nullptr) ||
           (metal_buffer != nullptr && segment_count <= 2);
  }
};

[[nodiscard]] inline MetalWindowDescriptor make_metal_window_descriptor(
    const MetalWindowSegmentPair& segments) {
  if (segments.segment_count() == 0) {
    return {};
  }
  return MetalWindowDescriptor{
      .metal_buffer = segments.first.present() ? segments.first.metal_buffer
                                               : segments.second.metal_buffer,
      .segment_count = static_cast<std::uint32_t>(segments.segment_count()),
      .first =
          MetalWindowRangeDescriptor{
              .offset_bytes = segments.first.offset_bytes,
              .size_bytes = segments.first.size_bytes(),
          },
      .second =
          MetalWindowRangeDescriptor{
              .offset_bytes = segments.second.offset_bytes,
              .size_bytes = segments.second.size_bytes(),
          },
  };
}

[[nodiscard]] inline MetalWindowDescriptor make_linear_metal_window_descriptor(
    const MetalBufferSlice& slice,
    std::size_t visible_bytes,
    std::size_t offset_bytes = 0) {
  if (!slice.valid() || offset_bytes > slice.writable_bytes.size()) {
    return {};
  }
  if (visible_bytes > slice.writable_bytes.size() - offset_bytes) {
    return {};
  }
  if (visible_bytes == 0) {
    return {};
  }
  return MetalWindowDescriptor{
      .metal_buffer = slice.metal_buffer,
      .segment_count = 1,
      .first =
          MetalWindowRangeDescriptor{
              .offset_bytes = offset_bytes,
              .size_bytes = visible_bytes,
          },
      .second = {},
  };
}

}  // namespace siliconrt
