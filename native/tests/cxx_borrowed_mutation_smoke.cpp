#include <cassert>
#include <cstdint>

#include "siliconrt/cxx_api.hpp"
#include "siliconrt/prefix_descriptor_builder.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("cxx-mut", 16384);

  auto budget = siliconrt::make_budget(prefix.total_bytes() * 2);
  auto arena = siliconrt::make_arena(prefix.total_bytes() * 2);
  auto handle =
      siliconrt::make_prefix_handle(arena.get(), budget.get(), prefix.as_c_descriptor());
  auto state =
      siliconrt::make_borrowed_decode_state(arena.get(), budget.get(), handle.get());

  siliconrt::set_decode_state_residency_promoting(
      state.get(),
      prefix.logical_token_count + 64,
      prefix.resident_token_count,
      prefix.sequence_bytes);

  siliconrt_decode_state_bindings_t bindings = {};
  assert(
      siliconrt_decode_state_describe_bindings(state.get(), &bindings) ==
      SILICONRT_STATUS_OK);
  assert(bindings.owns_sequence);
  assert(bindings.borrows_constant);

  return 0;
}
