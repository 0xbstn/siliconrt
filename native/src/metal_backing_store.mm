#include "siliconrt/metal_backing_store.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace siliconrt {

struct MetalBackingStoreBackend::Impl {
  struct StoreState {
    siliconrt_backing_store_descriptor_t descriptor = {};
    __strong id<MTLBuffer> buffer = nil;
  };

  Impl() : device(MTLCreateSystemDefaultDevice()) {
    if (!device) {
      throw std::runtime_error("MTLCreateSystemDefaultDevice failed");
    }
  }

  void materialize_store(const siliconrt_backing_store_descriptor_t& descriptor) {
    if (!descriptor.present || descriptor.backing_store_id == 0) {
      return;
    }

    auto it = stores.find(descriptor.backing_store_id);
    if (it != stores.end()) {
      const auto& existing = it->second.descriptor;
      if (existing.kind != descriptor.kind ||
          existing.capacity_bytes != descriptor.capacity_bytes ||
          existing.global_base_offset_bytes != descriptor.global_base_offset_bytes) {
        throw std::runtime_error("conflicting Metal backing store descriptor");
      }
      return;
    }

    id<MTLBuffer> buffer =
        [device newBufferWithLength:descriptor.capacity_bytes
                            options:MTLResourceStorageModeShared];
    if (!buffer) {
      throw std::runtime_error("newBufferWithLength failed");
    }

    stores.emplace(
        descriptor.backing_store_id,
        StoreState{
            .descriptor = descriptor,
            .buffer = buffer,
        });
  }

  __strong id<MTLDevice> device = nil;
  std::unordered_map<std::uint64_t, StoreState> stores;
};

MetalBackingStoreBackend::MetalBackingStoreBackend()
    : impl_(std::make_unique<Impl>()) {}

MetalBackingStoreBackend::~MetalBackingStoreBackend() = default;

MetalBackingStoreBackend::MetalBackingStoreBackend(
    MetalBackingStoreBackend&&) noexcept = default;

MetalBackingStoreBackend& MetalBackingStoreBackend::operator=(
    MetalBackingStoreBackend&&) noexcept = default;

void MetalBackingStoreBackend::materialize(const BackingStoreLayoutView& layout) {
  impl_->materialize_store(layout.raw.sequence);
  impl_->materialize_store(layout.raw.constant_state);
}

std::size_t MetalBackingStoreBackend::store_count() const {
  return impl_->stores.size();
}

bool MetalBackingStoreBackend::has_store(std::uint64_t backing_store_id) const {
  return impl_->stores.find(backing_store_id) != impl_->stores.end();
}

std::string MetalBackingStoreBackend::device_name() const {
  NSString* name = impl_->device.name;
  return name ? std::string(name.UTF8String) : std::string();
}

void* MetalBackingStoreBackend::metal_device() const {
  return (__bridge void*)impl_->device;
}

MetalBufferSlice MetalBackingStoreBackend::resolve(const BufferSliceView& view) {
  if (!view.valid()) {
    throw std::runtime_error("resolve called with invalid BufferSliceView");
  }

  auto it = impl_->stores.find(view.binding.backing_store_id);
  if (it == impl_->stores.end()) {
    throw std::runtime_error("Metal backing store not materialized");
  }

  const auto& state = it->second;
  if (state.descriptor.kind != view.binding.backing_store_kind) {
    throw std::runtime_error("Metal backing store kind mismatch");
  }
  if (state.descriptor.backing_store_id != view.binding.backing_store_id) {
    throw std::runtime_error("Metal backing store id mismatch");
  }
  if (view.store_relative_end() > state.descriptor.capacity_bytes) {
    throw std::runtime_error("Metal slice exceeds backing store capacity");
  }
  if (!view.matches_global_mapping()) {
    throw std::runtime_error("Metal slice does not match global mapping");
  }

  auto* base = static_cast<std::uint8_t*>(state.buffer.contents);
  if (!base) {
    throw std::runtime_error("MTLBuffer.contents returned null");
  }
  auto* begin = base + static_cast<std::ptrdiff_t>(view.store_relative_begin());

  return MetalBufferSlice{
      .view = view,
      .bytes = std::span<std::uint8_t>(
          begin, static_cast<std::size_t>(view.binding.used_bytes)),
      .writable_bytes = std::span<std::uint8_t>(
          begin, static_cast<std::size_t>(view.binding.capacity_bytes)),
      .metal_buffer = (__bridge void*)state.buffer,
  };
}

}  // namespace siliconrt
