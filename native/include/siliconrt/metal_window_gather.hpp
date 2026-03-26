#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_descriptor.hpp"

namespace siliconrt {

struct MetalWindowGatherKernelDescriptor {
  std::uint32_t segment_count = 0;
  std::uint32_t reserved = 0;
  std::uint64_t first_offset_bytes = 0;
  std::uint64_t first_size_bytes = 0;
  std::uint64_t second_offset_bytes = 0;
  std::uint64_t second_size_bytes = 0;
  std::uint64_t destination_offset_bytes = 0;
  std::uint64_t total_bytes = 0;
};

[[nodiscard]] inline MetalWindowGatherKernelDescriptor make_metal_window_gather_descriptor(
    const MetalWindowDescriptor& descriptor,
    std::size_t destination_offset_bytes = 0) {
  return MetalWindowGatherKernelDescriptor{
      .segment_count = descriptor.segment_count,
      .reserved = 0,
      .first_offset_bytes = descriptor.first.offset_bytes,
      .first_size_bytes = descriptor.first.size_bytes,
      .second_offset_bytes = descriptor.second.offset_bytes,
      .second_size_bytes = descriptor.second.size_bytes,
      .destination_offset_bytes = destination_offset_bytes,
      .total_bytes = descriptor.total_bytes(),
  };
}

class MetalWindowGather {
 public:
  explicit MetalWindowGather(const MetalBackingStoreBackend& backend);
  explicit MetalWindowGather(void* metal_device);
  ~MetalWindowGather();

  MetalWindowGather(MetalWindowGather&&) noexcept;
  MetalWindowGather& operator=(MetalWindowGather&&) noexcept;

  MetalWindowGather(const MetalWindowGather&) = delete;
  MetalWindowGather& operator=(const MetalWindowGather&) = delete;

  void gather(
      const MetalWindowDescriptor& source,
      const MetalBufferSlice& destination,
      std::size_t destination_offset_bytes = 0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
