#pragma once

#include <cstdint>

#include "siliconrt/c_api.h"

namespace siliconrt {

struct StorageBindingView {
  siliconrt_storage_binding_t raw = {};

  [[nodiscard]] constexpr bool present() const {
    return raw.present;
  }

  [[nodiscard]] constexpr bool owned() const {
    return raw.ownership == SILICONRT_STORAGE_OWNERSHIP_OWNED;
  }

  [[nodiscard]] constexpr bool borrowed() const {
    return raw.ownership == SILICONRT_STORAGE_OWNERSHIP_BORROWED;
  }

  [[nodiscard]] constexpr bool has_backing_store() const {
    return raw.backing_store_id != 0;
  }

  [[nodiscard]] constexpr std::uint64_t visible_bytes() const {
    return present() ? raw.used_bytes : 0;
  }

  [[nodiscard]] constexpr std::uint64_t owned_bytes() const {
    return owned() ? raw.used_bytes : 0;
  }

  [[nodiscard]] constexpr std::uint64_t borrowed_bytes() const {
    return borrowed() ? raw.used_bytes : 0;
  }
};

struct StorageLayoutView {
  siliconrt_storage_layout_t raw = {};

  [[nodiscard]] constexpr StorageBindingView sequence() const {
    return StorageBindingView{raw.sequence};
  }

  [[nodiscard]] constexpr StorageBindingView constant_state() const {
    return StorageBindingView{raw.constant_state};
  }

  [[nodiscard]] constexpr std::uint64_t visible_bytes() const {
    return sequence().visible_bytes() + constant_state().visible_bytes();
  }

  [[nodiscard]] constexpr std::uint64_t owned_bytes() const {
    return sequence().owned_bytes() + constant_state().owned_bytes();
  }

  [[nodiscard]] constexpr std::uint64_t borrowed_bytes() const {
    return sequence().borrowed_bytes() + constant_state().borrowed_bytes();
  }

  [[nodiscard]] constexpr bool borrows_any() const {
    return sequence().borrowed() || constant_state().borrowed();
  }
};

struct BackingStoreDescriptorView {
  siliconrt_backing_store_descriptor_t raw = {};

  [[nodiscard]] constexpr bool present() const {
    return raw.present;
  }
};

struct BackingStoreLayoutView {
  siliconrt_backing_store_layout_t raw = {};

  [[nodiscard]] constexpr BackingStoreDescriptorView sequence() const {
    return BackingStoreDescriptorView{raw.sequence};
  }

  [[nodiscard]] constexpr BackingStoreDescriptorView constant_state() const {
    return BackingStoreDescriptorView{raw.constant_state};
  }
};

struct StorageHandleDescriptorView {
  siliconrt_storage_handle_descriptor_t raw = {};

  [[nodiscard]] constexpr bool present() const {
    return raw.present;
  }
};

struct StorageHandleLayoutView {
  siliconrt_storage_handle_layout_t raw = {};

  [[nodiscard]] constexpr StorageHandleDescriptorView sequence() const {
    return StorageHandleDescriptorView{raw.sequence};
  }

  [[nodiscard]] constexpr StorageHandleDescriptorView constant_state() const {
    return StorageHandleDescriptorView{raw.constant_state};
  }
};

}  // namespace siliconrt
