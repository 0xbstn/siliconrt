#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/circular_sequence_plan.hpp"
#include "siliconrt/metal_backing_store.hpp"
#include "siliconrt/metal_bounded_sequence.hpp"
#include "siliconrt/metal_buffer_ops.hpp"
#include "siliconrt/metal_circular_sequence.hpp"
#include "siliconrt/metal_window_checksum.hpp"
#include "siliconrt/metal_window_linearizer.hpp"
#include "siliconrt/metal_window_segments.hpp"
#include "siliconrt/metal_window_stats.hpp"
#include "siliconrt/metal_window_score.hpp"
#include "siliconrt/metal_window_gather.hpp"
#include "siliconrt/storage_slice.hpp"

namespace {

struct BenchConfig {
  std::size_t sequence_capacity_bytes = 131072;
  std::size_t append_bytes = 4096;
  std::size_t trim_target_bytes = 65536;
  std::size_t iterations = 2000;
  bool json = false;
};

double per_iteration_ns(
    const std::chrono::steady_clock::duration total,
    const std::size_t iterations) {
  const auto total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
  return static_cast<double>(total_ns) / static_cast<double>(iterations);
}

std::size_t parse_size(const char* value) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == nullptr || *end != '\0') {
    throw std::runtime_error("invalid integer argument");
  }
  return static_cast<std::size_t>(parsed);
}

BenchConfig parse_args(int argc, char** argv) {
  BenchConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--capacity-bytes" && i + 1 < argc) {
      config.sequence_capacity_bytes = parse_size(argv[++i]);
    } else if (arg == "--append-bytes" && i + 1 < argc) {
      config.append_bytes = parse_size(argv[++i]);
    } else if (arg == "--trim-target-bytes" && i + 1 < argc) {
      config.trim_target_bytes = parse_size(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      config.iterations = parse_size(argv[++i]);
    } else if (arg == "--json") {
      config.json = true;
    } else {
      throw std::runtime_error("unknown or incomplete argument");
    }
  }

  if (config.sequence_capacity_bytes == 0 || config.iterations == 0) {
    throw std::runtime_error("capacity and iterations must be non-zero");
  }
  if (config.trim_target_bytes > config.sequence_capacity_bytes) {
    throw std::runtime_error("trim-target-bytes cannot exceed capacity-bytes");
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  const BenchConfig config = parse_args(argc, argv);
  const auto sequence_capacity_bytes = config.sequence_capacity_bytes;
  const auto append_bytes = config.append_bytes;
  const auto trim_target_bytes = config.trim_target_bytes;
  const auto iterations = config.iterations;

  auto budget = siliconrt::make_budget(sequence_capacity_bytes * 4);
  auto arena =
      siliconrt::make_partitioned_arena(sequence_capacity_bytes * 2, 4096);

  siliconrt_prefix_descriptor_t desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "metal-bounded-sequence-bench",
      .logical_token_count = 8192,
      .resident_token_count = 8192,
      .sequence_bytes = sequence_capacity_bytes,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };

  auto handle = siliconrt::make_prefix_handle(arena.get(), budget.get(), desc);
  siliconrt_prefix_descriptor_t linear_desc = {
      .model_key = "qwen35_9b_text",
      .prefix_hash_hex = "metal-bounded-sequence-linear",
      .logical_token_count = 8192,
      .resident_token_count = 8192,
      .sequence_bytes = sequence_capacity_bytes,
      .constant_bytes = 0,
      .cache_mode = SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS,
  };
  auto linear_handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), linear_desc);

  siliconrt::MetalBackingStoreBackend backend;
  const auto stores = siliconrt::describe_arena_backing_stores(arena.get());
  backend.materialize(stores);
  siliconrt::MetalBufferOps ops(backend);
  siliconrt::MetalBoundedSequence bounded(&ops);
  siliconrt::MetalCircularSequence circular(&ops);
  siliconrt::MetalWindowChecksum checksum(backend);
  siliconrt::MetalWindowStats stats(backend);
  siliconrt::MetalWindowScore score(backend);
  siliconrt::MetalWindowLinearizer linearizer(&ops);
  siliconrt::MetalWindowGather gather(backend);

  const siliconrt::StorageSliceLayoutView slices{
      .layout = siliconrt::describe_prefix_storage(handle.get()),
      .backing_stores = stores,
  };
  const siliconrt::StorageSliceLayoutView linear_slices{
      .layout = siliconrt::describe_prefix_storage(linear_handle.get()),
      .backing_stores = stores,
  };
  auto sequence = backend.resolve(slices.sequence());
  auto linear_sequence = backend.resolve(linear_slices.sequence());
  ops.fill(sequence, 0x11);
  ops.fill(linear_sequence, 0);

  std::vector<std::uint8_t> appended(append_bytes, 0x7B);

  siliconrt::CircularSequenceState circular_state{
      .head_offset_bytes = 0,
      .used_bytes = sequence_capacity_bytes,
      .capacity_bytes = sequence_capacity_bytes,
  };

  const auto append_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    bounded.append(
        sequence,
        sequence_capacity_bytes,
        sequence,
        std::span<const std::uint8_t>(appended.data(), appended.size()));
  }
  const auto append_end = std::chrono::steady_clock::now();

  const auto circular_append_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    circular_state =
        circular.append(
                    sequence,
                    circular_state,
                    std::span<const std::uint8_t>(appended.data(), appended.size()))
            .plan.after;
  }
  const auto circular_append_end = std::chrono::steady_clock::now();

  const auto linearize_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    linearizer.copy_to_linear(segments, linear_sequence);
  }
  const auto linearize_end = std::chrono::steady_clock::now();

  const auto gather_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    gather.gather(descriptor, linear_sequence);
  }
  const auto gather_end = std::chrono::steady_clock::now();

  std::uint64_t checksum_sink = 0;
  const auto checksum_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    checksum_sink ^= checksum.checksum(descriptor);
  }
  const auto checksum_end = std::chrono::steady_clock::now();

  std::uint64_t score_sink = 0;
  const auto score_direct_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    score_sink ^= score.score(descriptor);
  }
  const auto score_direct_end = std::chrono::steady_clock::now();

  const auto score_linearized_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    linearizer.copy_to_linear(segments, linear_sequence);
    score_sink ^= score.score(
        siliconrt::make_linear_metal_window_descriptor(linear_sequence, descriptor.total_bytes()));
  }
  const auto score_linearized_end = std::chrono::steady_clock::now();

  siliconrt::MetalWindowStatsResult stats_sink = {0, 0, 0, 0};
  const auto stats_direct_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    const auto result = stats.stats(descriptor);
    for (std::size_t j = 0; j < stats_sink.size(); ++j) {
      stats_sink[j] ^= result[j];
    }
  }
  const auto stats_direct_end = std::chrono::steady_clock::now();

  const auto stats_linearized_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto segments = siliconrt::make_metal_window_segments(sequence, circular_state);
    const auto descriptor = siliconrt::make_metal_window_descriptor(segments);
    linearizer.copy_to_linear(segments, linear_sequence);
    const auto result = stats.stats(
        siliconrt::make_linear_metal_window_descriptor(linear_sequence, descriptor.total_bytes()));
    for (std::size_t j = 0; j < stats_sink.size(); ++j) {
      stats_sink[j] ^= result[j];
    }
  }
  const auto stats_linearized_end = std::chrono::steady_clock::now();

  const auto trim_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    bounded.trim_front(sequence, sequence_capacity_bytes, trim_target_bytes, 0);
  }
  const auto trim_end = std::chrono::steady_clock::now();

  const auto bounded_append_ns =
      per_iteration_ns(append_end - append_start, iterations);
  const auto circular_append_ns =
      per_iteration_ns(circular_append_end - circular_append_start, iterations);
  const auto window_linearize_ns =
      per_iteration_ns(linearize_end - linearize_start, iterations);
  const auto window_gather_ns =
      per_iteration_ns(gather_end - gather_start, iterations);
  const auto window_checksum_ns =
      per_iteration_ns(checksum_end - checksum_start, iterations);
  const auto window_score_direct_ns =
      per_iteration_ns(score_direct_end - score_direct_start, iterations);
  const auto window_score_linearized_ns =
      per_iteration_ns(score_linearized_end - score_linearized_start, iterations);
  const auto window_stats_direct_ns =
      per_iteration_ns(stats_direct_end - stats_direct_start, iterations);
  const auto window_stats_linearized_ns =
      per_iteration_ns(stats_linearized_end - stats_linearized_start, iterations);
  const auto trim_front_ns =
      per_iteration_ns(trim_end - trim_start, iterations);

  if (config.json) {
    std::cout << "{"
              << "\"capacity_bytes\":" << sequence_capacity_bytes << ","
              << "\"append_bytes\":" << append_bytes << ","
              << "\"trim_target_bytes\":" << trim_target_bytes << ","
              << "\"iterations\":" << iterations << ","
              << "\"metal_bounded_append_ns\":" << bounded_append_ns << ","
              << "\"metal_circular_append_ns\":" << circular_append_ns << ","
              << "\"metal_window_linearize_ns\":" << window_linearize_ns << ","
              << "\"metal_window_gather_ns\":" << window_gather_ns << ","
              << "\"metal_window_checksum_ns\":" << window_checksum_ns << ","
              << "\"metal_window_score_direct_ns\":" << window_score_direct_ns << ","
              << "\"metal_window_score_linearized_ns\":" << window_score_linearized_ns << ","
              << "\"metal_window_stats_direct_ns\":" << window_stats_direct_ns << ","
              << "\"metal_window_stats_linearized_ns\":" << window_stats_linearized_ns << ","
              << "\"metal_trim_front_ns\":" << trim_front_ns << "}"
              << "\n";
  } else {
    std::cout << "metal_bounded_append_ns=" << bounded_append_ns << "\n";
    std::cout << "metal_circular_append_ns=" << circular_append_ns << "\n";
    std::cout << "metal_window_linearize_ns=" << window_linearize_ns << "\n";
    std::cout << "metal_window_gather_ns=" << window_gather_ns << "\n";
    std::cout << "metal_window_checksum_ns=" << window_checksum_ns << "\n";
    std::cout << "metal_window_score_direct_ns=" << window_score_direct_ns << "\n";
    std::cout << "metal_window_score_linearized_ns=" << window_score_linearized_ns << "\n";
    std::cout << "metal_window_stats_direct_ns=" << window_stats_direct_ns << "\n";
    std::cout << "metal_window_stats_linearized_ns=" << window_stats_linearized_ns << "\n";
    std::cout << "metal_trim_front_ns=" << trim_front_ns << "\n";
  }

  std::cerr << "checksum_sink=" << checksum_sink << " score_sink=" << score_sink
            << " stats_sink=" << stats_sink[0] << "," << stats_sink[1] << ","
            << stats_sink[2] << "," << stats_sink[3] << "\n";

  return 0;
}
