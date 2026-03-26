#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "siliconrt/circular_sequence_plan.hpp"
#include "siliconrt/metal_buffer_ops.hpp"

namespace siliconrt {

struct MetalCircularAppendResult {
  CircularAppendPlan plan = {};
};

class MetalCircularSequence {
 public:
  explicit MetalCircularSequence(MetalBufferOps* ops) : ops_(ops) {}

  MetalCircularAppendResult append(
      const MetalBufferSlice& slice,
      const CircularSequenceState& state,
      std::span<const std::uint8_t> appended_bytes) const;

 private:
  MetalBufferOps* ops_ = nullptr;
};

}  // namespace siliconrt
