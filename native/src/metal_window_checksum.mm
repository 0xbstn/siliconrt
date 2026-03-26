#include "siliconrt/metal_window_checksum.hpp"
#include "siliconrt/metal_compute_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

namespace siliconrt {

namespace {

struct WindowChecksumKernelDescriptor {
  std::uint32_t segment_count = 0;
  std::uint32_t reserved = 0;
  std::uint64_t first_offset_bytes = 0;
  std::uint64_t first_size_bytes = 0;
  std::uint64_t second_offset_bytes = 0;
  std::uint64_t second_size_bytes = 0;
};

static NSString* kWindowChecksumKernelSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct WindowDescriptor {
  uint segment_count;
  uint reserved;
  ulong first_offset_bytes;
  ulong first_size_bytes;
  ulong second_offset_bytes;
  ulong second_size_bytes;
};

kernel void siliconrt_window_checksum(
    device const uchar* src [[buffer(0)]],
    constant WindowDescriptor& desc [[buffer(1)]],
    device ulong* out_checksum [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
  ulong partial = 0;
  const ulong total_bytes = desc.first_size_bytes + desc.second_size_bytes;
  for (ulong index = tid; index < total_bytes; index += threads_per_group) {
    ulong source_offset = 0;
    if (index < desc.first_size_bytes) {
      source_offset = desc.first_offset_bytes + index;
    } else {
      source_offset = desc.second_offset_bytes + (index - desc.first_size_bytes);
    }
    partial += static_cast<ulong>(src[source_offset]);
  }

  threadgroup ulong shared[256];
  shared[tid] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    out_checksum[0] = shared[0];
  }
}
)METAL";

}  // namespace

struct MetalWindowChecksum::Impl {
  explicit Impl(void* opaque_device)
      : device((__bridge id<MTLDevice>)opaque_device),
        runtime(opaque_device, kWindowChecksumKernelSource.UTF8String, "siliconrt_window_checksum") {
    if (!device) {
      throw std::runtime_error("MetalWindowChecksum requires a valid MTLDevice");
    }
    result_buffer =
        [device newBufferWithLength:sizeof(std::uint64_t)
                            options:MTLResourceStorageModeShared];
    if (!result_buffer) {
      throw std::runtime_error("MetalWindowChecksum result buffer allocation failed");
    }
  }

  std::uint64_t submit_checksum(const MetalWindowDescriptor& source) const {
    if (!source.valid() || source.metal_buffer == nullptr) {
      throw std::runtime_error("checksum called with invalid source descriptor");
    }

    const WindowChecksumKernelDescriptor descriptor = {
        .segment_count = source.segment_count,
        .reserved = 0,
        .first_offset_bytes = source.first.offset_bytes,
        .first_size_bytes = source.first.size_bytes,
        .second_offset_bytes = source.second.offset_bytes,
        .second_size_bytes = source.second.size_bytes,
    };

    if (source.total_bytes() == 0) {
      return 0;
    }

    auto* result_ptr = static_cast<std::uint64_t*>(result_buffer.contents);
    if (result_ptr == nullptr) {
      throw std::runtime_error("MetalWindowChecksum result buffer contents failed");
    }
    *result_ptr = 0;

    id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.metal_buffer;
    if (!src) {
      throw std::runtime_error("MetalWindowChecksum called with null source buffer");
    }

    constexpr NSUInteger thread_count = 256;
    runtime.dispatch_threadgroups_1d(1, thread_count, [&](void* opaque_encoder) {
      id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)opaque_encoder;
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBytes:&descriptor length:sizeof(WindowChecksumKernelDescriptor) atIndex:1];
      [encoder setBuffer:result_buffer offset:0 atIndex:2];
    });

    return *result_ptr;
  }

  __strong id<MTLDevice> device = nil;
  MetalComputeRuntime runtime;
  __strong id<MTLBuffer> result_buffer = nil;
};

MetalWindowChecksum::MetalWindowChecksum(const MetalBackingStoreBackend& backend)
    : MetalWindowChecksum(backend.metal_device()) {}

MetalWindowChecksum::MetalWindowChecksum(void* metal_device)
    : impl_(std::make_unique<Impl>(metal_device)) {}

MetalWindowChecksum::~MetalWindowChecksum() = default;

MetalWindowChecksum::MetalWindowChecksum(MetalWindowChecksum&&) noexcept = default;

MetalWindowChecksum& MetalWindowChecksum::operator=(MetalWindowChecksum&&) noexcept = default;

std::uint64_t MetalWindowChecksum::checksum(const MetalWindowDescriptor& source) const {
  return impl_->submit_checksum(source);
}

}  // namespace siliconrt
