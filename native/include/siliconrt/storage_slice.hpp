#pragma once

#include <cstdint>

#include "siliconrt/storage_layout.hpp"

namespace siliconrt {

struct BufferSliceView {
  siliconrt_storage_binding_t binding = {};
  siliconrt_backing_store_descriptor_t backing_store = {};

  [[nodiscard]] constexpr bool valid() const {
    return binding.present && backing_store.present &&
           binding.backing_store_id == backing_store.backing_store_id;
  }

  [[nodiscard]] constexpr bool borrowed() const {
    return binding.ownership == SILICONRT_STORAGE_OWNERSHIP_BORROWED;
  }

  [[nodiscard]] constexpr bool owned() const {
    return binding.ownership == SILICONRT_STORAGE_OWNERSHIP_OWNED;
  }

  [[nodiscard]] constexpr std::uint64_t store_relative_begin() const {
    return binding.backing_store_offset_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t store_relative_end() const {
    return binding.backing_store_offset_bytes + binding.used_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t global_begin() const {
    return binding.offset_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t global_end() const {
    return binding.offset_bytes + binding.used_bytes;
  }

  [[nodiscard]] constexpr bool matches_global_mapping() const {
    return global_begin() ==
           backing_store.global_base_offset_bytes + store_relative_begin();
  }
};

struct StorageSliceLayoutView {
  StorageLayoutView layout = {};
  BackingStoreLayoutView backing_stores = {};

  [[nodiscard]] constexpr BufferSliceView sequence() const {
    return BufferSliceView{
        .binding = layout.raw.sequence,
        .backing_store = backing_stores.raw.sequence,
    };
  }

  [[nodiscard]] constexpr BufferSliceView constant_state() const {
    return BufferSliceView{
        .binding = layout.raw.constant_state,
        .backing_store = backing_stores.raw.constant_state,
    };
  }

  [[nodiscard]] constexpr std::uint64_t visible_bytes() const {
    return layout.visible_bytes();
  }
};

}  // namespace siliconrt
