#include "siliconrt/metal_compute_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

namespace siliconrt {

struct MetalComputeRuntime::Impl {
  Impl(void* opaque_device, const char* source, const char* function_name)
      : device((__bridge id<MTLDevice>)opaque_device) {
    if (!device) {
      throw std::runtime_error("MetalComputeRuntime requires a valid MTLDevice");
    }
    queue = [device newCommandQueue];
    if (!queue) {
      throw std::runtime_error("MetalComputeRuntime newCommandQueue failed");
    }

    NSError* error = nil;
    NSString* source_string = [NSString stringWithUTF8String:source];
    NSString* function_string = [NSString stringWithUTF8String:function_name];
    id<MTLLibrary> library =
        [device newLibraryWithSource:source_string options:nil error:&error];
    if (!library) {
      const auto message = error != nil ? std::string(error.localizedDescription.UTF8String)
                                        : std::string("unknown Metal compile error");
      throw std::runtime_error("MetalComputeRuntime shader compile failed: " + message);
    }

    id<MTLFunction> function = [library newFunctionWithName:function_string];
    if (!function) {
      throw std::runtime_error("MetalComputeRuntime function lookup failed");
    }

    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      const auto message = error != nil ? std::string(error.localizedDescription.UTF8String)
                                        : std::string("unknown pipeline error");
      throw std::runtime_error("MetalComputeRuntime pipeline creation failed: " + message);
    }
  }

  void dispatch_threadgroups_1d(
      std::size_t threadgroups_x,
      std::size_t threads_per_threadgroup_x,
      const std::function<void(void* encoder)>& encode) const {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (!command_buffer) {
      throw std::runtime_error("MetalComputeRuntime commandBuffer creation failed");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (!encoder) {
      throw std::runtime_error("MetalComputeRuntime computeCommandEncoder creation failed");
    }

    [encoder setComputePipelineState:pipeline];
    encode((__bridge void*)encoder);
    [encoder dispatchThreadgroups:MTLSizeMake(threadgroups_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                       threads_per_threadgroup_x == 0
                                           ? 1
                                           : threads_per_threadgroup_x,
                                       1,
                                       1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  void dispatch_threads_1d(
      std::size_t threads_x,
      std::size_t threads_per_threadgroup_x,
      const std::function<void(void* encoder)>& encode) const {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (!command_buffer) {
      throw std::runtime_error("MetalComputeRuntime commandBuffer creation failed");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (!encoder) {
      throw std::runtime_error("MetalComputeRuntime computeCommandEncoder creation failed");
    }

    [encoder setComputePipelineState:pipeline];
    encode((__bridge void*)encoder);
    [encoder dispatchThreads:MTLSizeMake(threads_x, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                 threads_per_threadgroup_x == 0
                                     ? 1
                                     : threads_per_threadgroup_x,
                                 1,
                                 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  __strong id<MTLDevice> device = nil;
  __strong id<MTLCommandQueue> queue = nil;
  __strong id<MTLComputePipelineState> pipeline = nil;
};

MetalComputeRuntime::MetalComputeRuntime(
    void* metal_device,
    const char* source,
    const char* function_name)
    : impl_(std::make_unique<Impl>(metal_device, source, function_name)) {}

MetalComputeRuntime::~MetalComputeRuntime() = default;

MetalComputeRuntime::MetalComputeRuntime(MetalComputeRuntime&&) noexcept = default;

MetalComputeRuntime& MetalComputeRuntime::operator=(MetalComputeRuntime&&) noexcept = default;

void MetalComputeRuntime::dispatch_threadgroups_1d(
    std::size_t threadgroups_x,
    std::size_t threads_per_threadgroup_x,
    const std::function<void(void* encoder)>& encode) const {
  impl_->dispatch_threadgroups_1d(threadgroups_x, threads_per_threadgroup_x, encode);
}

void MetalComputeRuntime::dispatch_threads_1d(
    std::size_t threads_x,
    std::size_t threads_per_threadgroup_x,
    const std::function<void(void* encoder)>& encode) const {
  impl_->dispatch_threads_1d(threads_x, threads_per_threadgroup_x, encode);
}

}  // namespace siliconrt
