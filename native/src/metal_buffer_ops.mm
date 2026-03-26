#include "siliconrt/metal_buffer_ops.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>

namespace siliconrt {

struct MetalBufferOps::Impl {
  explicit Impl(void* opaque_device)
      : device((__bridge id<MTLDevice>)opaque_device) {
    if (!device) {
      throw std::runtime_error("MetalBufferOps requires a valid MTLDevice");
    }
    queue = [device newCommandQueue];
    if (!queue) {
      throw std::runtime_error("newCommandQueue failed");
    }
  }

  void submit_fill(const MetalBufferSlice& slice, std::uint8_t value) {
    submit_fill_region(slice, 0, slice.bytes.size(), value);
  }

  void submit_fill_region(
      const MetalBufferSlice& slice,
      std::size_t destination_offset_bytes,
      std::size_t byte_count,
      std::uint8_t value) {
    if (!slice.valid()) {
      throw std::runtime_error("fill called with invalid MetalBufferSlice");
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)slice.metal_buffer;
    if (!buffer) {
      throw std::runtime_error("fill called with null MTLBuffer");
    }
    if (destination_offset_bytes > slice.writable_bytes.size()) {
      throw std::runtime_error("fill destination offset exceeds slice capacity");
    }
    if (byte_count > slice.writable_bytes.size() - destination_offset_bytes) {
      throw std::runtime_error("fill byte_count exceeds slice capacity");
    }
    if (byte_count == 0) {
      return;
    }

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (!command_buffer) {
      throw std::runtime_error("commandBuffer creation failed");
    }
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    if (!blit) {
      throw std::runtime_error("blitCommandEncoder creation failed");
    }

    [blit fillBuffer:buffer
               range:NSMakeRange(
                         static_cast<NSUInteger>(
                             slice.view.store_relative_begin() +
                             destination_offset_bytes),
                         static_cast<NSUInteger>(byte_count))
               value:value];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  void submit_copy(
      const MetalBufferSlice& source,
      const MetalBufferSlice& destination,
      std::size_t byte_count) {
    submit_copy_region(source, 0, destination, 0, byte_count);
  }

  void submit_copy_region(
      const MetalBufferSlice& source,
      std::size_t source_offset_bytes,
      const MetalBufferSlice& destination,
      std::size_t destination_offset_bytes,
      std::size_t byte_count) {
    if (!source.valid() || !destination.valid()) {
      throw std::runtime_error("copy called with invalid MetalBufferSlice");
    }

    id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.metal_buffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)destination.metal_buffer;
    if (!src || !dst) {
      throw std::runtime_error("copy called with null MTLBuffer");
    }

    const auto resolved_byte_count =
        byte_count == 0 ? source.bytes.size() : byte_count;
    if (source_offset_bytes > source.writable_bytes.size() ||
        destination_offset_bytes > destination.writable_bytes.size()) {
      throw std::runtime_error("copy offsets exceed slice size");
    }
    if (resolved_byte_count >
            source.writable_bytes.size() - source_offset_bytes ||
        resolved_byte_count >
            destination.writable_bytes.size() - destination_offset_bytes) {
      throw std::runtime_error("copy byte_count exceeds slice size");
    }
    if (resolved_byte_count == 0) {
      return;
    }
    if (src == dst) {
      auto* src_begin = source.writable_bytes.data() +
                        static_cast<std::ptrdiff_t>(source_offset_bytes);
      auto* dst_begin = destination.writable_bytes.data() +
                        static_cast<std::ptrdiff_t>(destination_offset_bytes);
      std::memmove(dst_begin, src_begin, resolved_byte_count);
      return;
    }

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (!command_buffer) {
      throw std::runtime_error("commandBuffer creation failed");
    }
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    if (!blit) {
      throw std::runtime_error("blitCommandEncoder creation failed");
    }

    [blit copyFromBuffer:src
            sourceOffset:static_cast<NSUInteger>(
                             source.view.store_relative_begin() + source_offset_bytes)
                toBuffer:dst
       destinationOffset:static_cast<NSUInteger>(
                             destination.view.store_relative_begin() +
                             destination_offset_bytes)
                    size:static_cast<NSUInteger>(resolved_byte_count)];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  __strong id<MTLDevice> device = nil;
  __strong id<MTLCommandQueue> queue = nil;
};

MetalBufferOps::MetalBufferOps(const MetalBackingStoreBackend& backend)
    : MetalBufferOps(backend.metal_device()) {}

MetalBufferOps::MetalBufferOps(void* metal_device)
    : impl_(std::make_unique<Impl>(metal_device)) {}

MetalBufferOps::~MetalBufferOps() = default;

MetalBufferOps::MetalBufferOps(MetalBufferOps&&) noexcept = default;

MetalBufferOps& MetalBufferOps::operator=(MetalBufferOps&&) noexcept = default;

void MetalBufferOps::fill(const MetalBufferSlice& slice, std::uint8_t value) {
  impl_->submit_fill(slice, value);
}

void MetalBufferOps::fill_region(
    const MetalBufferSlice& slice,
    std::size_t destination_offset_bytes,
    std::size_t byte_count,
    std::uint8_t value) {
  impl_->submit_fill_region(slice, destination_offset_bytes, byte_count, value);
}

void MetalBufferOps::copy(
    const MetalBufferSlice& source,
    const MetalBufferSlice& destination,
    std::size_t byte_count) {
  impl_->submit_copy(source, destination, byte_count);
}

void MetalBufferOps::copy_region(
    const MetalBufferSlice& source,
    std::size_t source_offset_bytes,
    const MetalBufferSlice& destination,
    std::size_t destination_offset_bytes,
    std::size_t byte_count) {
  impl_->submit_copy_region(
      source,
      source_offset_bytes,
      destination,
      destination_offset_bytes,
      byte_count);
}

}  // namespace siliconrt
