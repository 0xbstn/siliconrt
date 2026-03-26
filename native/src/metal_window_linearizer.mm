#include "siliconrt/metal_window_linearizer.hpp"

#include <stdexcept>

namespace siliconrt {

void MetalWindowLinearizer::copy_to_linear(
    const MetalWindowSegmentPair& segments,
    const MetalBufferSlice& destination,
    std::size_t destination_offset_bytes) const {
  if (ops_ == nullptr) {
    throw std::runtime_error("MetalWindowLinearizer requires MetalBufferOps");
  }
  if (!destination.valid()) {
    throw std::runtime_error("copy_to_linear called with invalid destination slice");
  }
  if (destination_offset_bytes > destination.writable_bytes.size()) {
    throw std::runtime_error("destination offset exceeds destination capacity");
  }
  if (segments.total_bytes() > destination.writable_bytes.size() - destination_offset_bytes) {
    throw std::runtime_error("destination capacity too small for window linearization");
  }

  std::size_t cursor = destination_offset_bytes;
  if (segments.first.present()) {
    if (!segments.first.source_slice.valid()) {
      throw std::runtime_error("first segment has invalid source slice");
    }
    ops_->copy_region(
        segments.first.source_slice,
        segments.first.offset_bytes,
        destination,
        cursor,
        segments.first.size_bytes());
    cursor += segments.first.size_bytes();
  }
  if (segments.second.present()) {
    if (!segments.second.source_slice.valid()) {
      throw std::runtime_error("second segment has invalid source slice");
    }
    ops_->copy_region(
        segments.second.source_slice,
        segments.second.offset_bytes,
        destination,
        cursor,
        segments.second.size_bytes());
  }
}

}  // namespace siliconrt
