#include <cassert>
#include <cstdint>

#include "siliconrt/window_policy.hpp"

int main() {
  constexpr auto keep_all = siliconrt::WindowPolicy::keep_all();
  constexpr auto fixed = siliconrt::WindowPolicy::fixed(2048);

  assert(keep_all.kind() == siliconrt::WindowPolicyKind::kKeepAll);
  assert(!keep_all.fixed_window_tokens().has_value());
  assert(keep_all.resident_tokens(8192) == 8192);
  assert(!keep_all.is_saturated(8192));

  assert(fixed.kind() == siliconrt::WindowPolicyKind::kFixedWindow);
  assert(fixed.fixed_window_tokens().has_value());
  assert(*fixed.fixed_window_tokens() == 2048);
  assert(fixed.resident_tokens(1024) == 1024);
  assert(fixed.resident_tokens(4096) == 2048);
  assert(!fixed.is_saturated(1024));
  assert(fixed.is_saturated(2048));

  return 0;
}
