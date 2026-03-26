#pragma once

#include <algorithm>
#include <cstdint>

#include "siliconrt/cache_profile.hpp"

namespace siliconrt {

struct SessionCapacityEstimate {
  std::uint64_t sequence_bytes = 0;
  std::uint64_t constant_bytes = 0;
  std::uint64_t total_bytes = 0;
};

struct PartitionedRuntimePlan {
  std::uint64_t total_capacity_bytes = 0;
  std::uint64_t sequence_capacity_bytes = 0;
  std::uint64_t constant_capacity_bytes = 0;
  std::uint64_t slack_bytes = 0;
  std::uint32_t window_tokens = 0;
  std::uint32_t target_sessions = 0;
  std::uint32_t max_sessions_by_sequence = 0;
  std::uint32_t max_sessions_by_constant = 0;
  std::uint32_t max_sessions_effective = 0;
  SessionCapacityEstimate per_session = {};

  [[nodiscard]] constexpr bool feasible() const {
    return max_sessions_effective >= target_sessions;
  }
};

[[nodiscard]] constexpr SessionCapacityEstimate estimate_session_capacity(
    const CacheProfile& profile,
    std::uint32_t window_tokens) {
  const auto footprint = profile.footprint(window_tokens, window_tokens);
  return SessionCapacityEstimate{
      .sequence_bytes = footprint.sequence_bytes,
      .constant_bytes = footprint.constant_bytes,
      .total_bytes = footprint.total_bytes(),
  };
}

[[nodiscard]] constexpr std::uint32_t capacity_to_sessions(
    std::uint64_t capacity_bytes,
    std::uint64_t bytes_per_session) {
  if (bytes_per_session == 0) {
    return 0;
  }
  return static_cast<std::uint32_t>(capacity_bytes / bytes_per_session);
}

[[nodiscard]] constexpr PartitionedRuntimePlan make_sequence_biased_plan(
    const CacheProfile& profile,
    std::uint32_t window_tokens,
    std::uint64_t total_capacity_bytes,
    std::uint32_t target_sessions) {
  const auto per_session = estimate_session_capacity(profile, window_tokens);
  const auto required_constant_bytes =
      per_session.constant_bytes * static_cast<std::uint64_t>(target_sessions);
  const auto required_sequence_bytes =
      per_session.sequence_bytes * static_cast<std::uint64_t>(target_sessions);
  const auto required_total_bytes =
      required_constant_bytes + required_sequence_bytes;

  PartitionedRuntimePlan plan;
  plan.total_capacity_bytes = total_capacity_bytes;
  plan.window_tokens = window_tokens;
  plan.target_sessions = target_sessions;
  plan.per_session = per_session;

  if (target_sessions == 0) {
    plan.sequence_capacity_bytes = total_capacity_bytes;
    plan.max_sessions_by_sequence =
        capacity_to_sessions(total_capacity_bytes, per_session.sequence_bytes);
    plan.max_sessions_effective = plan.max_sessions_by_sequence;
    return plan;
  }

  plan.constant_capacity_bytes =
      std::min(required_constant_bytes, total_capacity_bytes);
  plan.sequence_capacity_bytes =
      total_capacity_bytes - plan.constant_capacity_bytes;
  plan.slack_bytes = total_capacity_bytes > required_total_bytes
      ? total_capacity_bytes - required_total_bytes
      : 0;
  plan.max_sessions_by_sequence =
      capacity_to_sessions(plan.sequence_capacity_bytes, per_session.sequence_bytes);
  plan.max_sessions_by_constant =
      capacity_to_sessions(plan.constant_capacity_bytes, per_session.constant_bytes);
  plan.max_sessions_effective = std::min(
      plan.max_sessions_by_sequence, plan.max_sessions_by_constant);
  return plan;
}

}  // namespace siliconrt
