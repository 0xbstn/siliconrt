#include "siliconrt/metal_window_gather.hpp"
#include "siliconrt/metal_compute_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>

namespace siliconrt {

namespace {

static NSString* kWindowGatherKernelSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct WindowGatherDescriptor {
  uint segment_count;
  uint reserved;
  ulong first_offset_bytes;
  ulong first_size_bytes;
  ulong second_offset_bytes;
  ulong second_size_bytes;
  ulong destination_offset_bytes;
  ulong total_bytes;
};

kernel void siliconrt_window_gather(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant WindowGatherDescriptor& desc [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  constexpr ulong kBytesPerThread = 16;
  const ulong base_index = static_cast<ulong>(gid) * kBytesPerThread;
  if (base_index >= desc.total_bytes) {
    return;
  }

  const ulong work_bytes =
      min(kBytesPerThread, desc.total_bytes - base_index);
  for (ulong lane = 0; lane < work_bytes; ++lane) {
    const ulong index = base_index + lane;
    ulong source_offset = 0;
    if (index < desc.first_size_bytes) {
      source_offset = desc.first_offset_bytes + index;
    } else {
      source_offset = desc.second_offset_bytes + (index - desc.first_size_bytes);
    }
    dst[desc.destination_offset_bytes + index] = src[source_offset];
  }
}
)METAL";

}  // namespace

struct MetalWindowGather::Impl {
  explicit Impl(void* opaque_device)
      : device((__bridge id<MTLDevice>)opaque_device),
        runtime(opaque_device, kWindowGatherKernelSource.UTF8String, "siliconrt_window_gather") {
    if (!device) {
      throw std::runtime_error("MetalWindowGather requires a valid MTLDevice");
    }
  }

  void submit_gather(
      const MetalWindowDescriptor& source,
      const MetalBufferSlice& destination,
      std::size_t destination_offset_bytes) const {
    if (!source.valid() || source.metal_buffer == nullptr) {
      throw std::runtime_error("gather called with invalid source descriptor");
    }
    if (!destination.valid()) {
      throw std::runtime_error("gather called with invalid destination slice");
    }

    const auto descriptor =
        make_metal_window_gather_descriptor(source, destination_offset_bytes);
    if (descriptor.total_bytes == 0) {
      return;
    }
    if (destination_offset_bytes > destination.writable_bytes.size() ||
        descriptor.total_bytes >
            destination.writable_bytes.size() - destination_offset_bytes) {
      throw std::runtime_error("gather exceeds destination slice capacity");
    }

    id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.metal_buffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)destination.metal_buffer;
    if (!src || !dst) {
      throw std::runtime_error("gather called with null MTLBuffer");
    }

    constexpr NSUInteger bytes_per_thread = 16;
    const NSUInteger total_threads =
        static_cast<NSUInteger>(
            (descriptor.total_bytes + bytes_per_thread - 1) / bytes_per_thread);
    const NSUInteger threads_per_group = total_threads < 256 ? total_threads : 256;
    runtime.dispatch_threads_1d(
        total_threads,
        threads_per_group == 0 ? 1 : threads_per_group,
        [&](void* opaque_encoder) {
          id<MTLComputeCommandEncoder> encoder =
              (__bridge id<MTLComputeCommandEncoder>)opaque_encoder;
          [encoder setBuffer:src offset:0 atIndex:0];
          [encoder setBuffer:dst offset:destination.view.store_relative_begin() atIndex:1];
          [encoder setBytes:&descriptor
                     length:sizeof(MetalWindowGatherKernelDescriptor)
                    atIndex:2];
        });
  }

  __strong id<MTLDevice> device = nil;
  MetalComputeRuntime runtime;
};

MetalWindowGather::MetalWindowGather(const MetalBackingStoreBackend& backend)
    : MetalWindowGather(backend.metal_device()) {}

MetalWindowGather::MetalWindowGather(void* metal_device)
    : impl_(std::make_unique<Impl>(metal_device)) {}

MetalWindowGather::~MetalWindowGather() = default;

MetalWindowGather::MetalWindowGather(MetalWindowGather&&) noexcept = default;

MetalWindowGather& MetalWindowGather::operator=(MetalWindowGather&&) noexcept = default;

void MetalWindowGather::gather(
    const MetalWindowDescriptor& source,
    const MetalBufferSlice& destination,
    std::size_t destination_offset_bytes) const {
  impl_->submit_gather(source, destination, destination_offset_bytes);
}

}  // namespace siliconrt
