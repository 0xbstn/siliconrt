#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_window_descriptor.hpp"

namespace siliconrt {

using MetalWindowStatsResult = std::array<std::uint64_t, 4>;

class MetalWindowStats {
 public:
  explicit MetalWindowStats(const MetalBackingStoreBackend& backend);
  explicit MetalWindowStats(void* metal_device);
  ~MetalWindowStats();

  MetalWindowStats(MetalWindowStats&&) noexcept;
  MetalWindowStats& operator=(MetalWindowStats&&) noexcept;

  MetalWindowStats(const MetalWindowStats&) = delete;
  MetalWindowStats& operator=(const MetalWindowStats&) = delete;

  [[nodiscard]] MetalWindowStatsResult stats(const MetalWindowDescriptor& source) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
