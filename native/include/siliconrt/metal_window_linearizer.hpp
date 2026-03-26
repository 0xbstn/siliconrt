#pragma once

#include <cstddef>

#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/metal_window_segments.hpp"

namespace siliconrt {

class MetalWindowLinearizer {
 public:
  explicit MetalWindowLinearizer(MetalBufferOps* ops) : ops_(ops) {}

  void copy_to_linear(
      const MetalWindowSegmentPair& segments,
      const MetalBufferSlice& destination,
      std::size_t destination_offset_bytes = 0) const;

 private:
  MetalBufferOps* ops_ = nullptr;
};

}  // namespace siliconrt
