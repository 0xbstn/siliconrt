#pragma once

#include <stdexcept>

#include "siliconrt/c_api.h"
#include "siliconrt/prefix_descriptor_builder.hpp"

namespace siliconrt {

class BoundedDecodeSession {
 public:
  BoundedDecodeSession(
      PrefixDescriptorBuilder builder,
      siliconrt_decode_state_t* state)
      : builder_(std::move(builder)), state_(state) {}

  BoundedDecodeSession(const BoundedDecodeSession&) = delete;
  BoundedDecodeSession& operator=(const BoundedDecodeSession&) = delete;

  BoundedDecodeSession(BoundedDecodeSession&& other) noexcept
      : builder_(std::move(other.builder_)), state_(other.state_) {
    other.state_ = nullptr;
  }

  BoundedDecodeSession& operator=(BoundedDecodeSession&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    builder_ = std::move(other.builder_);
    state_ = other.state_;
    other.state_ = nullptr;
    return *this;
  }

  [[nodiscard]] siliconrt_decode_state_t* release() {
    auto* out = state_;
    state_ = nullptr;
    return out;
  }

  [[nodiscard]] siliconrt_decode_state_t* state() const { return state_; }

  [[nodiscard]] OwnedPrefixDescriptor descriptor() const {
    siliconrt_prefix_descriptor_t out = {};
    const auto status = siliconrt_decode_state_describe(state_, &out);
    if (status != SILICONRT_STATUS_OK) {
      throw std::runtime_error("decode_state_describe failed");
    }
    return OwnedPrefixDescriptor{
        .model_key = out.model_key,
        .prefix_hash_hex = out.prefix_hash_hex,
        .logical_token_count = out.logical_token_count,
        .resident_token_count = out.resident_token_count,
        .sequence_bytes = out.sequence_bytes,
        .constant_bytes = out.constant_bytes,
        .cache_mode = out.cache_mode,
    };
  }

  [[nodiscard]] ResidencyDelta append_tokens(std::uint32_t appended_tokens) {
    const auto current = descriptor();
    const auto next = builder_.advance(current, appended_tokens);
    const auto delta = builder_.delta_after_append(current, appended_tokens);
    const auto status = siliconrt_decode_state_set_residency(
        state_,
        next.logical_token_count,
        next.resident_token_count,
        next.sequence_bytes);
    if (status != SILICONRT_STATUS_OK) {
      throw std::runtime_error("decode_state_set_residency failed");
    }
    return delta;
  }

 private:
  PrefixDescriptorBuilder builder_;
  siliconrt_decode_state_t* state_ = nullptr;
};

}  // namespace siliconrt
