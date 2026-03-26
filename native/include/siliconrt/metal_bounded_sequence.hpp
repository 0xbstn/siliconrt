#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "siliconrt/bounded_sequence_plan.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/trim_plan.hpp"

namespace siliconrt {

struct MetalBoundedAppendResult {
  BoundedAppendPlan plan = {};
};

struct MetalTrimFrontResult {
  TrimFrontPlan plan = {};
};

class MetalBoundedSequence {
 public:
  explicit MetalBoundedSequence(MetalBufferOps* ops) : ops_(ops) {}

  MetalBoundedAppendResult append(
      const MetalBufferSlice& source,
      std::size_t current_used_bytes,
      const MetalBufferSlice& destination,
      std::span<const std::uint8_t> appended_bytes) const;
  MetalTrimFrontResult trim_front(
      const MetalBufferSlice& slice,
      std::size_t current_used_bytes,
      std::size_t target_used_bytes,
      std::uint8_t fill_value = 0) const;

 private:
  MetalBufferOps* ops_ = nullptr;
};

}  // namespace siliconrt
