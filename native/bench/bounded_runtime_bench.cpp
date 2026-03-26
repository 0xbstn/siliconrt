#include <chrono>
#include <cstdint>
#include <iostream>

#include "siliconrt/bounded_runtime.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::BoundedRuntime runtime(
      siliconrt::PrefixDescriptorBuilder(planner),
      siliconrt::profiles::qwen35_9b_text().footprint(16384, 2048).total_bytes() * 2);

  const auto descriptor = runtime.make_prefix_descriptor("bench-prefix", 16384);
  auto handle = runtime.materialize_prefix(descriptor);
  auto session = runtime.restore_decode_session(handle.get());

  constexpr std::uint32_t kSteps = 100000;
  constexpr std::uint32_t kAppendPerStep = 1;

  const auto start = std::chrono::steady_clock::now();
  std::uint64_t additional_bytes = 0;
  for (std::uint32_t i = 0; i < kSteps; ++i) {
    const auto delta = session.append_tokens(kAppendPerStep);
    additional_bytes += delta.additional_total_bytes;
  }
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const auto ns_per_step = static_cast<double>(elapsed_ns) / kSteps;
  const auto final_descriptor = session.descriptor();

  std::cout << "{\n";
  std::cout << "  \"bench\": \"bounded_runtime_append\",\n";
  std::cout << "  \"steps\": " << kSteps << ",\n";
  std::cout << "  \"append_per_step\": " << kAppendPerStep << ",\n";
  std::cout << "  \"elapsed_ns\": " << elapsed_ns << ",\n";
  std::cout << "  \"ns_per_step\": " << ns_per_step << ",\n";
  std::cout << "  \"final_logical_tokens\": "
            << final_descriptor.logical_token_count << ",\n";
  std::cout << "  \"final_resident_tokens\": "
            << final_descriptor.resident_token_count << ",\n";
  std::cout << "  \"additional_total_bytes\": " << additional_bytes << "\n";
  std::cout << "}\n";

  siliconrt_decode_state_destroy(
      runtime.arena(), runtime.budget(), session.release());
  return 0;
}
