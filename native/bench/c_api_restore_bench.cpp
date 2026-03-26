#include <chrono>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "siliconrt/c_api.h"

namespace {

double run_clone_all(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_prefix_handle_t* handle,
    std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    siliconrt_decode_state_t* state = nullptr;
    if (siliconrt_decode_restore(arena, budget, handle, &state) !=
        SILICONRT_STATUS_OK) {
      std::cerr << "decode_restore failed\n";
      std::exit(1);
    }
    siliconrt_decode_state_destroy(arena, budget, state);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(elapsed_ns) / static_cast<double>(iterations);
}

double run_borrowed(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_prefix_handle_t* handle,
    std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    siliconrt_decode_state_t* state = nullptr;
    if (siliconrt_decode_restore_borrowed(arena, budget, handle, &state) !=
        SILICONRT_STATUS_OK) {
      std::cerr << "decode_restore_borrowed failed\n";
      std::exit(1);
    }
    siliconrt_decode_state_destroy(arena, budget, state);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(elapsed_ns) / static_cast<double>(iterations);
}

double run_borrowed_mutating(
    siliconrt_arena_t* arena,
    siliconrt_budget_t* budget,
    siliconrt_prefix_handle_t* handle,
    const siliconrt_prefix_descriptor_t& prefix,
    std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < iterations; ++i) {
    siliconrt_decode_state_t* state = nullptr;
    if (siliconrt_decode_restore_borrowed(arena, budget, handle, &state) !=
        SILICONRT_STATUS_OK) {
      std::cerr << "decode_restore_borrowed failed\n";
      std::exit(1);
    }
    if (siliconrt_decode_state_set_residency_promoting(
            state,
            prefix.logical_token_count + 64,
            prefix.resident_token_count,
            prefix.sequence_bytes) != SILICONRT_STATUS_OK) {
      std::cerr << "decode_state_set_residency_promoting failed\n";
      std::exit(1);
    }
    siliconrt_decode_state_destroy(arena, budget, state);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(elapsed_ns) / static_cast<double>(iterations);
}

}  // namespace

int main() {
  const siliconrt_prefix_descriptor_t prefix = {
      .model_key = "qwen35_9b",
      .prefix_hash_hex = "bench-c-api",
      .logical_token_count = 16384,
      .resident_token_count = 2048,
      .sequence_bytes = 67108864ULL,
      .constant_bytes = 26345472ULL,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  const auto total_capacity =
      prefix.sequence_bytes * 10 + prefix.constant_bytes * 2;
  constexpr std::uint64_t iterations = 100000;

  auto run_mode = [&](bool partitioned) {
    siliconrt_budget_t* budget = nullptr;
    siliconrt_arena_t* arena = nullptr;
    if (siliconrt_budget_create(total_capacity, &budget) != SILICONRT_STATUS_OK) {
      std::cerr << "budget_create failed\n";
      std::exit(1);
    }
    const auto arena_status = partitioned
        ? siliconrt_arena_create_partitioned(
              prefix.sequence_bytes * 10,
              prefix.constant_bytes * 2,
              &arena)
        : siliconrt_arena_create(total_capacity, &arena);
    if (arena_status != SILICONRT_STATUS_OK) {
      std::cerr << "arena_create failed\n";
      std::exit(1);
    }

    siliconrt_prefix_handle_t* handle = nullptr;
    if (siliconrt_prefix_create(arena, budget, &prefix, &handle) !=
        SILICONRT_STATUS_OK) {
      std::cerr << "prefix_create failed\n";
      std::exit(1);
    }

    const auto clone_all_ns = run_clone_all(arena, budget, handle, iterations);
    const auto borrowed_ns = run_borrowed(arena, budget, handle, iterations);
    const auto borrowed_mutating_ns =
        run_borrowed_mutating(arena, budget, handle, prefix, iterations);

    siliconrt_prefix_destroy(arena, budget, handle);
    siliconrt_arena_destroy(arena);
    siliconrt_budget_destroy(budget);

    return std::tuple<double, double, double>{
        clone_all_ns, borrowed_ns, borrowed_mutating_ns};
  };

  const auto [unified_clone_all_ns, unified_borrowed_ns, unified_borrowed_mutating_ns] =
      run_mode(false);
  const auto [partitioned_clone_all_ns, partitioned_borrowed_ns,
              partitioned_borrowed_mutating_ns] = run_mode(true);

  std::cout << "{\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"unified_clone_all_restore_destroy_ns_per_iteration\": "
            << unified_clone_all_ns << ",\n"
            << "  \"unified_borrowed_restore_destroy_ns_per_iteration\": "
            << unified_borrowed_ns << ",\n"
            << "  \"unified_borrowed_mutating_restore_destroy_ns_per_iteration\": "
            << unified_borrowed_mutating_ns << ",\n"
            << "  \"partitioned_clone_all_restore_destroy_ns_per_iteration\": "
            << partitioned_clone_all_ns << ",\n"
            << "  \"partitioned_borrowed_restore_destroy_ns_per_iteration\": "
            << partitioned_borrowed_ns << ",\n"
            << "  \"partitioned_borrowed_mutating_restore_destroy_ns_per_iteration\": "
            << partitioned_borrowed_mutating_ns << ",\n"
            << "  \"sequence_bytes\": " << prefix.sequence_bytes << ",\n"
            << "  \"constant_bytes\": " << prefix.constant_bytes << "\n"
            << "}\n";
  return 0;
}
