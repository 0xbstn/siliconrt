#include <cassert>
#include <cstdint>

#include "siliconrt/cache_profile.hpp"

int main() {
  constexpr auto profile = siliconrt::profiles::qwen35_9b_text();

  static_assert(profile.sequence_layer_count == 8);
  static_assert(profile.constant_layer_count == 24);
  static_assert(profile.sequence_bytes_per_token == 32768);
  static_assert(profile.constant_bytes == 26345472);

  const auto unbounded_8192 = profile.footprint(8192);
  assert(unbounded_8192.logical_tokens == 8192);
  assert(unbounded_8192.resident_tokens == 8192);
  assert(unbounded_8192.sequence_bytes == 268435456);
  assert(unbounded_8192.constant_bytes == 26345472);
  assert(unbounded_8192.total_bytes() == 294780928);

  const auto bounded_8192_4096 = profile.footprint(8192, 4096);
  assert(bounded_8192_4096.resident_tokens == 4096);
  assert(bounded_8192_4096.sequence_bytes == 134217728);
  assert(bounded_8192_4096.total_bytes() == 160563200);
  assert(bounded_8192_4096.saved_bytes_vs(unbounded_8192) == 134217728);

  const auto unbounded_16384 = profile.footprint(16384);
  assert(unbounded_16384.total_bytes() == 563216384);

  const auto bounded_16384_2048 = profile.footprint(16384, 2048);
  assert(bounded_16384_2048.resident_tokens == 2048);
  assert(bounded_16384_2048.sequence_bytes == 67108864);
  assert(bounded_16384_2048.total_bytes() == 93454336);
  assert(bounded_16384_2048.saved_bytes_vs(unbounded_16384) == 469762048);

  return 0;
}
