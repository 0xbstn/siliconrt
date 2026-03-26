#include "siliconrt/metal_window_stats.hpp"
#include "siliconrt/metal_compute_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

namespace siliconrt {

namespace {

struct WindowStatsKernelDescriptor {
  std::uint32_t segment_count = 0;
  std::uint32_t reserved = 0;
  std::uint64_t first_offset_bytes = 0;
  std::uint64_t first_size_bytes = 0;
  std::uint64_t second_offset_bytes = 0;
  std::uint64_t second_size_bytes = 0;
};

static NSString* kWindowStatsKernelSource = @R"METAL(
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

kernel void siliconrt_window_stats(
    device const uchar* src [[buffer(0)]],
    constant WindowDescriptor& desc [[buffer(1)]],
    device ulong4* out_stats [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
  ulong4 partial = ulong4(0);
  const ulong total_bytes = desc.first_size_bytes + desc.second_size_bytes;
  for (ulong index = tid; index < total_bytes; index += threads_per_group) {
    ulong source_offset = 0;
    if (index < desc.first_size_bytes) {
      source_offset = desc.first_offset_bytes + index;
    } else {
      source_offset = desc.second_offset_bytes + (index - desc.first_size_bytes);
    }

    const ulong value = static_cast<ulong>(src[source_offset]);
    const ulong w0 = (index & 31ul) + 1ul;
    const ulong w1 = ((index >> 3) & 15ul) + 3ul;
    partial.x += value;
    partial.y += value * value;
    partial.z += value * w0;
    partial.w += value * value * w1;
  }

  threadgroup ulong4 shared[256];
  shared[tid] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    out_stats[0] = shared[0];
  }
}
)METAL";

}  // namespace

struct MetalWindowStats::Impl {
  explicit Impl(void* opaque_device)
      : device((__bridge id<MTLDevice>)opaque_device),
        runtime(opaque_device, kWindowStatsKernelSource.UTF8String, "siliconrt_window_stats") {
    if (!device) {
      throw std::runtime_error("MetalWindowStats requires a valid MTLDevice");
    }
    result_buffer =
        [device newBufferWithLength:sizeof(MetalWindowStatsResult)
                            options:MTLResourceStorageModeShared];
    if (!result_buffer) {
      throw std::runtime_error("MetalWindowStats result buffer allocation failed");
    }
  }

  MetalWindowStatsResult submit_stats(const MetalWindowDescriptor& source) const {
    if (!source.valid() || source.metal_buffer == nullptr) {
      throw std::runtime_error("stats called with invalid source descriptor");
    }

    const WindowStatsKernelDescriptor descriptor = {
        .segment_count = source.segment_count,
        .reserved = 0,
        .first_offset_bytes = source.first.offset_bytes,
        .first_size_bytes = source.first.size_bytes,
        .second_offset_bytes = source.second.offset_bytes,
        .second_size_bytes = source.second.size_bytes,
    };

    if (source.total_bytes() == 0) {
      return {};
    }

    auto* result_ptr = static_cast<MetalWindowStatsResult*>(result_buffer.contents);
    if (result_ptr == nullptr) {
      throw std::runtime_error("MetalWindowStats result buffer contents failed");
    }
    *result_ptr = {};

    id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.metal_buffer;
    if (!src) {
      throw std::runtime_error("MetalWindowStats called with null source buffer");
    }

    constexpr NSUInteger thread_count = 256;
    runtime.dispatch_threadgroups_1d(1, thread_count, [&](void* opaque_encoder) {
      id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)opaque_encoder;
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBytes:&descriptor length:sizeof(WindowStatsKernelDescriptor) atIndex:1];
      [encoder setBuffer:result_buffer offset:0 atIndex:2];
    });

    return *result_ptr;
  }

  __strong id<MTLDevice> device = nil;
  MetalComputeRuntime runtime;
  __strong id<MTLBuffer> result_buffer = nil;
};

MetalWindowStats::MetalWindowStats(const MetalBackingStoreBackend& backend)
    : MetalWindowStats(backend.metal_device()) {}

MetalWindowStats::MetalWindowStats(void* metal_device)
    : impl_(std::make_unique<Impl>(metal_device)) {}

MetalWindowStats::~MetalWindowStats() = default;

MetalWindowStats::MetalWindowStats(MetalWindowStats&&) noexcept = default;

MetalWindowStats& MetalWindowStats::operator=(MetalWindowStats&&) noexcept = default;

MetalWindowStatsResult MetalWindowStats::stats(const MetalWindowDescriptor& source) const {
  return impl_->submit_stats(source);
}

}  // namespace siliconrt
