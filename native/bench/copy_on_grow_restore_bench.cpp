#include <chrono>
#include <cstdint>
#include <iostream>

#include "siliconrt/partitioned_prefix_store.hpp"

namespace {

double run_borrow_release(
    siliconrt::PartitionedPrefixStore& store,
    std::uint64_t prefix_handle_id,
    std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    auto decode = store.restore_borrow_until_append(prefix_handle_id);
    if (!decode.has_value()) {
      std::cerr << "restore_borrow_until_append failed\n";
      std::exit(1);
    }
    if (!store.release_decode(decode->handle_id)) {
      std::cerr << "release_decode failed\n";
      std::exit(1);
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(elapsed_ns) / static_cast<double>(iterations);
}

double run_borrow_promote_release(
    siliconrt::PartitionedPrefixStore& store,
    std::uint64_t prefix_handle_id,
    std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    auto decode = store.restore_borrow_until_append(prefix_handle_id);
    if (!decode.has_value()) {
      std::cerr << "restore_borrow_until_append failed\n";
      std::exit(1);
    }
    if (!store.promote_decode_sequence(decode->handle_id)) {
      std::cerr << "promote_decode_sequence failed\n";
      std::exit(1);
    }
    if (!store.release_decode(decode->handle_id)) {
      std::cerr << "release_decode failed\n";
      std::exit(1);
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(elapsed_ns) / static_cast<double>(iterations);
}

}  // namespace

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("bench-copy-on-grow", 16384);
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
  const auto borrow_release_ns =
      run_borrow_release(store, handle->handle_id, iterations);
  const auto borrow_promote_release_ns =
      run_borrow_promote_release(store, handle->handle_id, iterations);

  std::cout << "{\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"borrow_release_ns_per_iteration\": " << borrow_release_ns << ",\n"
            << "  \"borrow_promote_release_ns_per_iteration\": "
            << borrow_promote_release_ns << ",\n"
            << "  \"prefix_bytes\": " << prefix.total_bytes() << ",\n"
            << "  \"sequence_promotion_bytes\": " << prefix.sequence_bytes << "\n"
            << "}\n";
  return 0;
}
