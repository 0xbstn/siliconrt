#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "siliconrt/storage_slice.hpp"

namespace siliconrt {

struct MetalBufferSlice {
  BufferSliceView view = {};
  std::span<std::uint8_t> bytes = {};
  std::span<std::uint8_t> writable_bytes = {};
  void* metal_buffer = nullptr;

  [[nodiscard]] bool valid() const {
    return view.valid() && bytes.size() == view.binding.used_bytes &&
           writable_bytes.size() == view.binding.capacity_bytes &&
           bytes.size() <= writable_bytes.size() && metal_buffer != nullptr;
  }
};

class MetalBackingStoreBackend {
 public:
  MetalBackingStoreBackend();
  ~MetalBackingStoreBackend();

  MetalBackingStoreBackend(MetalBackingStoreBackend&&) noexcept;
  MetalBackingStoreBackend& operator=(MetalBackingStoreBackend&&) noexcept;

  MetalBackingStoreBackend(const MetalBackingStoreBackend&) = delete;
  MetalBackingStoreBackend& operator=(const MetalBackingStoreBackend&) = delete;

  void materialize(const BackingStoreLayoutView& layout);

  [[nodiscard]] std::size_t store_count() const;
  [[nodiscard]] bool has_store(std::uint64_t backing_store_id) const;
  [[nodiscard]] std::string device_name() const;
  [[nodiscard]] void* metal_device() const;

  [[nodiscard]] MetalBufferSlice resolve(const BufferSliceView& view);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
