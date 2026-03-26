#include <cassert>

#include "siliconrt/trim_plan.hpp"

int main() {
  const auto same = siliconrt::make_trim_front_plan(16, 32);
  assert(!same.trims());
  assert(same.kept_bytes == 16);
  assert(same.trimmed_bytes == 0);
  assert(same.source_keep_offset_bytes == 0);

  const auto trim = siliconrt::make_trim_front_plan(16, 6);
  assert(trim.trims());
  assert(trim.kept_bytes == 6);
  assert(trim.trimmed_bytes == 10);
  assert(trim.source_keep_offset_bytes == 10);
  assert(trim.destination_used_bytes == 6);

  const auto empty = siliconrt::make_trim_front_plan(16, 0);
  assert(empty.trims());
  assert(empty.kept_bytes == 0);
  assert(empty.trimmed_bytes == 16);
  assert(empty.source_keep_offset_bytes == 16);

  return 0;
}
