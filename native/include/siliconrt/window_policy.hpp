#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

namespace siliconrt {

enum class WindowPolicyKind : std::uint8_t {
  kKeepAll = 0,
  kFixedWindow = 1,
};

class WindowPolicy {
 public:
  [[nodiscard]] static constexpr WindowPolicy keep_all() {
    return WindowPolicy(WindowPolicyKind::kKeepAll, 0);
  }

  [[nodiscard]] static constexpr WindowPolicy fixed(std::uint32_t window_tokens) {
    return WindowPolicy(WindowPolicyKind::kFixedWindow, window_tokens);
  }

  [[nodiscard]] constexpr WindowPolicyKind kind() const { return kind_; }

  [[nodiscard]] constexpr std::optional<std::uint32_t> fixed_window_tokens() const {
    if (kind_ != WindowPolicyKind::kFixedWindow) {
      return std::nullopt;
    }
    return window_tokens_;
  }

  [[nodiscard]] constexpr std::uint32_t resident_tokens(
      std::uint32_t logical_tokens) const {
    if (kind_ != WindowPolicyKind::kFixedWindow) {
      return logical_tokens;
    }
    return std::min(logical_tokens, window_tokens_);
  }

  [[nodiscard]] constexpr bool is_saturated(
      std::uint32_t resident_tokens) const {
    return kind_ == WindowPolicyKind::kFixedWindow &&
           resident_tokens >= window_tokens_;
  }

 private:
  constexpr WindowPolicy(
      WindowPolicyKind kind,
      std::uint32_t window_tokens)
      : kind_(kind), window_tokens_(window_tokens) {}

  WindowPolicyKind kind_ = WindowPolicyKind::kKeepAll;
  std::uint32_t window_tokens_ = 0;
};

}  // namespace siliconrt
