#include <cassert>
#include <cstdint>

#include "siliconrt/prefix_descriptor_builder.hpp"
#include "siliconrt/restore_plan.hpp"

int main() {
  constexpr siliconrt::BoundedWindowPlanner planner(
      siliconrt::profiles::qwen35_9b_text(), 2048);
  siliconrt::PrefixDescriptorBuilder builder(planner);
  const auto prefix = builder.make_prefix("restore-a", 16384);

  const auto conservative = siliconrt::RestorePlanner::make_plan(
      prefix, siliconrt::RestoreAliasMode::kCloneAll);
  assert(conservative.clone_sequence_bytes == 67108864ULL);
  assert(conservative.clone_constant_bytes == 26345472ULL);
  assert(conservative.additional_bytes() == prefix.total_bytes());
  assert(conservative.visible_bytes() == prefix.total_bytes());

  const auto share_constant = siliconrt::RestorePlanner::make_plan(
      prefix, siliconrt::RestoreAliasMode::kShareConstantState);
  assert(share_constant.clone_sequence_bytes == 67108864ULL);
  assert(share_constant.clone_constant_bytes == 0);
  assert(share_constant.borrowed_constant_bytes == 26345472ULL);
  assert(share_constant.additional_bytes() == 67108864ULL);
  assert(share_constant.visible_bytes() == prefix.total_bytes());

  const auto borrow_both = siliconrt::RestorePlanner::make_plan(
      prefix, siliconrt::RestoreAliasMode::kBorrowSequenceAndConstant);
  assert(borrow_both.clone_sequence_bytes == 0);
  assert(borrow_both.clone_constant_bytes == 0);
  assert(borrow_both.borrowed_sequence_bytes == 67108864ULL);
  assert(borrow_both.borrowed_constant_bytes == 26345472ULL);
  assert(borrow_both.additional_bytes() == 0);
  assert(borrow_both.visible_bytes() == prefix.total_bytes());
  assert(borrow_both.requires_sequence_promotion());

  return 0;
}
