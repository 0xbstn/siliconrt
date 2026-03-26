#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace siliconrt {

class MetalComputeRuntime {
 public:
  MetalComputeRuntime(
      void* metal_device,
      const char* source,
      const char* function_name);
  ~MetalComputeRuntime();

  MetalComputeRuntime(MetalComputeRuntime&&) noexcept;
  MetalComputeRuntime& operator=(MetalComputeRuntime&&) noexcept;

  MetalComputeRuntime(const MetalComputeRuntime&) = delete;
  MetalComputeRuntime& operator=(const MetalComputeRuntime&) = delete;

  void dispatch_threadgroups_1d(
      std::size_t threadgroups_x,
      std::size_t threads_per_threadgroup_x,
      const std::function<void(void* encoder)>& encode) const;

  void dispatch_threads_1d(
      std::size_t threads_x,
      std::size_t threads_per_threadgroup_x,
      const std::function<void(void* encoder)>& encode) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace siliconrt
