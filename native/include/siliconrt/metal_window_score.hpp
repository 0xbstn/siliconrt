#pragma once

#include <cstdint>
#include <memory>

#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_descriptor.hpp"

namespace siliconrt {

class MetalWindowScore {
 public:
  explicit MetalWindowScore(const MetalBackingStoreBackend& backend);
  explicit MetalWindowScore(void* metal_device);
  ~MetalWindowScore();

  MetalWindowScore(MetalWindowScore&&) noexcept;
  MetalWindowScore& operator=(MetalWindowScore&&) noexcept;

  MetalWindowScore(const MetalWindowScore&) = delete;
  MetalWindowScore& operator=(const MetalWindowScore&) = delete;

  [[nodiscard]] std::uint64_t score(const MetalWindowDescriptor& source) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
