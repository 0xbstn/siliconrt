#pragma once

#include <cstdint>

namespace siliconrt {

class KvBudget {
 public:
  explicit KvBudget(std::uint64_t capacity_bytes)
      : capacity_bytes_(capacity_bytes) {}

  [[nodiscard]] std::uint64_t capacity_bytes() const { return capacity_bytes_; }
  [[nodiscard]] std::uint64_t reserved_bytes() const { return reserved_bytes_; }
  [[nodiscard]] std::uint64_t committed_bytes() const { return committed_bytes_; }
  [[nodiscard]] std::uint64_t used_bytes() const {
    return reserved_bytes_ + committed_bytes_;
  }
  [[nodiscard]] std::uint64_t available_bytes() const {
    return capacity_bytes_ - used_bytes();
  }

  [[nodiscard]] bool can_reserve(std::uint64_t bytes) const {
    return bytes <= available_bytes();
  }

  bool reserve(std::uint64_t bytes) {
    if (!can_reserve(bytes)) {
      return false;
    }
    reserved_bytes_ += bytes;
    return true;
  }

  bool commit_reserved(std::uint64_t bytes) {
    if (bytes > reserved_bytes_) {
      return false;
    }
    reserved_bytes_ -= bytes;
    committed_bytes_ += bytes;
    return true;
  }

  bool commit_direct(std::uint64_t bytes) {
    if (!can_reserve(bytes)) {
      return false;
    }
    committed_bytes_ += bytes;
    return true;
  }

  bool release_reserved(std::uint64_t bytes) {
    if (bytes > reserved_bytes_) {
      return false;
    }
    reserved_bytes_ -= bytes;
    return true;
  }

  bool release_committed(std::uint64_t bytes) {
    if (bytes > committed_bytes_) {
      return false;
    }
    committed_bytes_ -= bytes;
    return true;
  }

  void reset() {
    reserved_bytes_ = 0;
    committed_bytes_ = 0;
  }

 private:
  std::uint64_t capacity_bytes_ = 0;
  std::uint64_t reserved_bytes_ = 0;
  std::uint64_t committed_bytes_ = 0;
};

}  // namespace siliconrt
