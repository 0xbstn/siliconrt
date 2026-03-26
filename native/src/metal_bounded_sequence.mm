#include "siliconrt/metal_bounded_sequence.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace siliconrt {

MetalBoundedAppendResult MetalBoundedSequence::append(
    const MetalBufferSlice& source,
    std::size_t current_used_bytes,
    const MetalBufferSlice& destination,
    std::span<const std::uint8_t> appended_bytes) const {
  if (ops_ == nullptr) {
    throw std::runtime_error("MetalBoundedSequence requires MetalBufferOps");
  }
  if (!source.valid() || !destination.valid()) {
    throw std::runtime_error("append called with invalid MetalBufferSlice");
  }
  if (current_used_bytes > source.writable_bytes.size()) {
    throw std::runtime_error("current_used_bytes exceeds source slice");
  }
  if (destination.view.binding.capacity_bytes > destination.writable_bytes.size()) {
    throw std::runtime_error("destination writable capacity is too small");
  }

  const auto plan = make_bounded_append_plan(
      current_used_bytes,
      appended_bytes.size(),
      destination.view.binding.capacity_bytes);

  if (plan.requires_tail_copy()) {
    ops_->copy_region(
        source,
        plan.source_keep_offset_bytes,
        destination,
        0,
        plan.kept_source_bytes);
  }

  if (plan.append_bytes != 0) {
    const auto* append_begin =
        appended_bytes.data() + static_cast<std::ptrdiff_t>(plan.append_source_offset_bytes);
    std::copy_n(
        append_begin,
        plan.append_bytes,
        destination.writable_bytes.begin() +
            static_cast<std::ptrdiff_t>(plan.destination_append_offset_bytes));
  }

  return MetalBoundedAppendResult{.plan = plan};
}

MetalTrimFrontResult MetalBoundedSequence::trim_front(
    const MetalBufferSlice& slice,
    std::size_t current_used_bytes,
    std::size_t target_used_bytes,
    std::uint8_t fill_value) const {
  if (ops_ == nullptr) {
    throw std::runtime_error("MetalBoundedSequence requires MetalBufferOps");
  }
  if (!slice.valid()) {
    throw std::runtime_error("trim_front called with invalid MetalBufferSlice");
  }
  if (current_used_bytes > slice.writable_bytes.size()) {
    throw std::runtime_error("current_used_bytes exceeds slice capacity");
  }

  const auto plan = make_trim_front_plan(current_used_bytes, target_used_bytes);

  if (plan.trims() && plan.kept_bytes != 0) {
    // On shared MTLBuffer storage, same-slice compaction is cheaper with a
    // direct host-visible memmove than with a queue roundtrip.
    auto* base = slice.writable_bytes.data();
    std::memmove(
        base,
        base + static_cast<std::ptrdiff_t>(plan.source_keep_offset_bytes),
        plan.kept_bytes);
  }
  if (plan.trimmed_bytes != 0) {
    std::fill_n(
        slice.writable_bytes.begin() +
            static_cast<std::ptrdiff_t>(plan.kept_bytes),
        plan.trimmed_bytes,
        fill_value);
  }

  return MetalTrimFrontResult{.plan = plan};
}

}  // namespace siliconrt
