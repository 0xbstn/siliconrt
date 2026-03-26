#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "siliconrt/metal_backing_store.hpp"

namespace siliconrt {

class MetalBufferOps {
 public:
  explicit MetalBufferOps(const MetalBackingStoreBackend& backend);
  explicit MetalBufferOps(void* metal_device);
  ~MetalBufferOps();

  MetalBufferOps(MetalBufferOps&&) noexcept;
  MetalBufferOps& operator=(MetalBufferOps&&) noexcept;

  MetalBufferOps(const MetalBufferOps&) = delete;
  MetalBufferOps& operator=(const MetalBufferOps&) = delete;

  void fill(const MetalBufferSlice& slice, std::uint8_t value);
  void fill_region(
      const MetalBufferSlice& slice,
      std::size_t destination_offset_bytes,
      std::size_t byte_count,
      std::uint8_t value);
  void copy(
      const MetalBufferSlice& source,
      const MetalBufferSlice& destination,
      std::size_t byte_count = 0);
  void copy_region(
      const MetalBufferSlice& source,
      std::size_t source_offset_bytes,
      const MetalBufferSlice& destination,
      std::size_t destination_offset_bytes,
      std::size_t byte_count);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
