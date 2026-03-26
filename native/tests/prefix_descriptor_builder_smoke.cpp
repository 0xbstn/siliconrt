#include <cassert>
#include <cstdint>

#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);

  const auto prefix = builder.make_prefix("prefix-a", 16384);
  assert(prefix.model_key == "qwen35_9b_text");
  assert(prefix.prefix_hash_hex == "prefix-a");
  assert(prefix.logical_token_count == 16384);
  assert(prefix.resident_token_count == 2048);
  assert(prefix.sequence_bytes == 67108864);
  assert(prefix.constant_bytes == 26345472);
  assert(prefix.total_bytes() == 93454336);

  const auto next = builder.advance(prefix, 64);
  assert(next.prefix_hash_hex == "prefix-a");
  assert(next.logical_token_count == 16448);
  assert(next.resident_token_count == 2048);
  assert(next.sequence_bytes == 67108864);
  assert(next.constant_bytes == 26345472);

  const auto delta = builder.delta_after_append(prefix, 64);
  assert(delta.appended_tokens == 64);
  assert(delta.additional_sequence_bytes == 0);
  assert(delta.additional_total_bytes == 0);

  const auto c_desc = next.as_c_descriptor();
  assert(c_desc.model_key != nullptr);
  assert(c_desc.prefix_hash_hex != nullptr);
  assert(c_desc.logical_token_count == 16448);
  assert(c_desc.resident_token_count == 2048);
  assert(c_desc.sequence_bytes == 67108864);
  assert(c_desc.constant_bytes == 26345472);
  assert(c_desc.cache_mode == SILICONRT_CACHE_MODE_BOUNDED_CONTIGUOUS);

  return 0;
}
