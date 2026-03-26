#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "siliconrt/storage_slice.hpp"

namespace siliconrt {

struct HostBufferSlice {
  BufferSliceView view = {};
  std::span<std::uint8_t> bytes = {};

  [[nodiscard]] bool valid() const {
    return view.valid() && bytes.size() == view.binding.used_bytes;
  }
};

class HostBackingStoreBackend {
 public:
  void materialize(const BackingStoreLayoutView& layout) {
    materialize_store(layout.raw.sequence);
    materialize_store(layout.raw.constant_state);
  }

  [[nodiscard]] std::size_t store_count() const {
    return stores_.size();
  }

  [[nodiscard]] bool has_store(std::uint64_t backing_store_id) const {
    return stores_.find(backing_store_id) != stores_.end();
  }

  [[nodiscard]] HostBufferSlice resolve(const BufferSliceView& view) {
    if (!view.valid()) {
      throw std::runtime_error("resolve called with invalid BufferSliceView");
    }

    auto it = stores_.find(view.binding.backing_store_id);
    if (it == stores_.end()) {
      throw std::runtime_error("backing store not materialized");
    }

    auto& state = it->second;
    if (state.descriptor.kind != view.binding.backing_store_kind) {
      throw std::runtime_error("backing store kind mismatch");
    }
    if (state.descriptor.backing_store_id != view.binding.backing_store_id) {
      throw std::runtime_error("backing store id mismatch");
    }
    if (view.store_relative_end() > state.bytes.size()) {
      throw std::runtime_error("slice exceeds backing store capacity");
    }
    if (!view.matches_global_mapping()) {
      throw std::runtime_error("slice does not match global mapping");
    }

    auto* begin = state.bytes.data() + static_cast<std::ptrdiff_t>(view.store_relative_begin());
    return HostBufferSlice{
        .view = view,
        .bytes = std::span<std::uint8_t>(
            begin, static_cast<std::size_t>(view.binding.used_bytes)),
    };
  }

 private:
  struct StoreState {
    siliconrt_backing_store_descriptor_t descriptor = {};
    std::vector<std::uint8_t> bytes = {};
  };

  void materialize_store(const siliconrt_backing_store_descriptor_t& descriptor) {
    if (!descriptor.present || descriptor.backing_store_id == 0) {
      return;
    }

    auto [it, inserted] = stores_.try_emplace(
        descriptor.backing_store_id,
        StoreState{
            .descriptor = descriptor,
            .bytes = std::vector<std::uint8_t>(
                static_cast<std::size_t>(descriptor.capacity_bytes), 0),
        });
    if (inserted) {
      return;
    }

    const auto& existing = it->second.descriptor;
    if (existing.kind != descriptor.kind ||
        existing.capacity_bytes != descriptor.capacity_bytes ||
        existing.global_base_offset_bytes != descriptor.global_base_offset_bytes) {
      throw std::runtime_error("conflicting backing store descriptor");
    }
  }

  std::unordered_map<std::uint64_t, StoreState> stores_;
};

}  // namespace siliconrt
