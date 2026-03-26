#include <chrono>
#include <cstdint>
#include <iostream>

#include "siliconrt/partitioned_prefix_store.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("bench-restore", 16384);
  const auto capacity_bytes =
      prefix.total_bytes() + (8 * prefix.sequence_bytes);
  const auto plan = siliconrt::make_sequence_biased_plan(
      planner.profile(), 2048, capacity_bytes, 1);
  siliconrt::PartitionedPrefixStore store(plan);
  auto handle = store.materialize(prefix);
  if (!handle.has_value()) {
    std::cerr << "failed to materialize prefix\n";
    return 1;
  }

  constexpr std::uint64_t iterations = 100000;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    auto decode = store.restore_share_constant(handle->handle_id);
    if (!decode.has_value()) {
      std::cerr << "restore_share_constant failed\n";
      return 1;
    }
    if (!store.release_decode(decode->handle_id)) {
      std::cerr << "release_decode failed\n";
      return 1;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              end - start)
                              .count();
  const auto ns_per_iteration =
      static_cast<double>(elapsed_ns) / static_cast<double>(iterations);

  std::cout << "{\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"elapsed_ns\": " << elapsed_ns << ",\n"
            << "  \"ns_per_iteration\": " << ns_per_iteration << ",\n"
            << "  \"prefix_bytes\": " << prefix.total_bytes() << ",\n"
            << "  \"additional_decode_bytes\": " << prefix.sequence_bytes << "\n"
            << "}\n";
  return 0;
}
