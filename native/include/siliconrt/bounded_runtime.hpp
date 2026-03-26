#pragma once

#include <cstdint>
#include <string>

#include "siliconrt/bounded_decode_session.hpp"
#include "siliconrt/cxx_api.hpp"

namespace siliconrt {

class BoundedRuntime {
 public:
  BoundedRuntime(PrefixDescriptorBuilder builder, std::uint64_t capacity_bytes)
      : builder_(std::move(builder)),
        budget_(make_budget(capacity_bytes)),
        arena_(make_arena(capacity_bytes)) {}

  [[nodiscard]] siliconrt_budget_t* budget() const { return budget_.get(); }
  [[nodiscard]] siliconrt_arena_t* arena() const { return arena_.get(); }
  [[nodiscard]] const PrefixDescriptorBuilder& builder() const { return builder_; }

  [[nodiscard]] OwnedPrefixDescriptor make_prefix_descriptor(
      std::string prefix_hash_hex,
      std::uint32_t logical_token_count) const {
    return builder_.make_prefix(std::move(prefix_hash_hex), logical_token_count);
  }

  [[nodiscard]] UniquePrefixHandle materialize_prefix(
      const OwnedPrefixDescriptor& descriptor) {
    const auto c_descriptor = descriptor.as_c_descriptor();
    return make_prefix_handle(arena_.get(), budget_.get(), c_descriptor);
  }

  [[nodiscard]] UniqueDecodeState restore_decode_state(
      const siliconrt_prefix_handle_t* handle) {
    return make_decode_state(arena_.get(), budget_.get(), handle);
  }

  [[nodiscard]] BoundedDecodeSession restore_decode_session(
      const siliconrt_prefix_handle_t* handle) {
    auto state = restore_decode_state(handle);
    return BoundedDecodeSession(builder_, state.release());
  }

 private:
  PrefixDescriptorBuilder builder_;
  UniqueBudget budget_;
  UniqueArena arena_;
};

}  // namespace siliconrt
