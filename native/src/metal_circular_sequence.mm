#include "siliconrt/metal_circular_sequence.hpp"

#include <algorithm>
#include <stdexcept>

namespace siliconrt {

MetalCircularAppendResult MetalCircularSequence::append(
    const MetalBufferSlice& slice,
    const CircularSequenceState& state,
    std::span<const std::uint8_t> appended_bytes) const {
  if (ops_ == nullptr) {
    throw std::runtime_error("MetalCircularSequence requires MetalBufferOps");
  }
  if (!slice.valid()) {
    throw std::runtime_error("append called with invalid MetalBufferSlice");
  }
  if (!state.valid()) {
    throw std::runtime_error("append called with invalid CircularSequenceState");
  }
  if (state.capacity_bytes > slice.writable_bytes.size()) {
    throw std::runtime_error("circular state capacity exceeds slice capacity");
  }

  const auto plan = make_circular_append_plan(state, appended_bytes.size());
  if (plan.append_bytes == 0) {
    return MetalCircularAppendResult{.plan = plan};
  }

  const auto* append_begin =
      appended_bytes.data() + static_cast<std::ptrdiff_t>(plan.append_source_offset_bytes);

  if (plan.append_segments.first.present()) {
    std::copy_n(
        append_begin,
        plan.append_segments.first.size_bytes,
        slice.writable_bytes.begin() +
            static_cast<std::ptrdiff_t>(plan.append_segments.first.offset_bytes));
  }
  if (plan.append_segments.second.present()) {
    std::copy_n(
        append_begin + static_cast<std::ptrdiff_t>(plan.append_segments.first.size_bytes),
        plan.append_segments.second.size_bytes,
        slice.writable_bytes.begin() +
            static_cast<std::ptrdiff_t>(plan.append_segments.second.offset_bytes));
  }

  return MetalCircularAppendResult{.plan = plan};
}

}  // namespace siliconrt
