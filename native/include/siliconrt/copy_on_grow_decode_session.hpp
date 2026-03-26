#pragma once

#include <stdexcept>

#include "siliconrt/partitioned_prefix_store.hpp"

namespace siliconrt {

class CopyOnGrowDecodeSession {
 public:
  CopyOnGrowDecodeSession(
      PartitionedPrefixStore* store,
      std::uint64_t handle_id)
      : store_(store), handle_id_(handle_id) {}

  CopyOnGrowDecodeSession(const CopyOnGrowDecodeSession&) = delete;
  CopyOnGrowDecodeSession& operator=(const CopyOnGrowDecodeSession&) = delete;

  CopyOnGrowDecodeSession(CopyOnGrowDecodeSession&& other) noexcept
      : store_(other.store_), handle_id_(other.handle_id_) {
    other.store_ = nullptr;
    other.handle_id_ = 0;
  }

  CopyOnGrowDecodeSession& operator=(CopyOnGrowDecodeSession&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    store_ = other.store_;
    handle_id_ = other.handle_id_;
    other.store_ = nullptr;
    other.handle_id_ = 0;
    return *this;
  }

  [[nodiscard]] std::uint64_t handle_id() const { return handle_id_; }

  [[nodiscard]] AliasedDecodeHandle describe() const {
    ensure_valid();
    const auto decode = store_->get_decode(handle_id_);
    if (!decode.has_value()) {
      throw std::runtime_error("get_decode failed");
    }
    return *decode;
  }

  [[nodiscard]] bool owns_sequence() const {
    return describe().owns_sequence();
  }

  bool promote_sequence() {
    ensure_valid();
    return store_->promote_decode_sequence(handle_id_);
  }

  bool release() {
    ensure_valid();
    const auto ok = store_->release_decode(handle_id_);
    if (ok) {
      store_ = nullptr;
      handle_id_ = 0;
    }
    return ok;
  }

 private:
  void ensure_valid() const {
    if (store_ == nullptr || handle_id_ == 0) {
      throw std::runtime_error("invalid copy-on-grow decode session");
    }
  }

  PartitionedPrefixStore* store_ = nullptr;
  std::uint64_t handle_id_ = 0;
};

}  // namespace siliconrt
