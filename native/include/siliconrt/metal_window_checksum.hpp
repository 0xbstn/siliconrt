#pragma once

#include <cstdint>
#include <memory>

#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_descriptor.hpp"

namespace siliconrt {

class MetalWindowChecksum {
 public:
  explicit MetalWindowChecksum(const MetalBackingStoreBackend& backend);
  explicit MetalWindowChecksum(void* metal_device);
  ~MetalWindowChecksum();

  MetalWindowChecksum(MetalWindowChecksum&&) noexcept;
  MetalWindowChecksum& operator=(MetalWindowChecksum&&) noexcept;

  MetalWindowChecksum(const MetalWindowChecksum&) = delete;
  MetalWindowChecksum& operator=(const MetalWindowChecksum&) = delete;

  [[nodiscard]] std::uint64_t checksum(const MetalWindowDescriptor& source) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
