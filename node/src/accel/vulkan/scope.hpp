#pragma once

#include "adapter/api.hpp"
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct ScopedBuffer {
  VulkanAdapter* adapter = nullptr;
  VulkanBuffer buffer{};
  VkDeviceSize used_bytes = 0u;

  ScopedBuffer() = default;
  ScopedBuffer(VulkanAdapter& owner, VulkanBuffer value, VkDeviceSize used = 0u)
      : adapter(&owner), buffer(value), used_bytes(used) {}
  ScopedBuffer(const ScopedBuffer&) = delete;
  ScopedBuffer& operator=(const ScopedBuffer&) = delete;
  ScopedBuffer(ScopedBuffer&& other) noexcept
      : adapter(other.adapter),
        buffer(other.buffer),
        used_bytes(other.used_bytes) {
    other.adapter = nullptr;
    other.buffer = VulkanBuffer{};
    other.used_bytes = 0u;
  }
  ScopedBuffer& operator=(ScopedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      adapter = other.adapter;
      buffer = other.buffer;
      used_bytes = other.used_bytes;
      other.adapter = nullptr;
      other.buffer = VulkanBuffer{};
      other.used_bytes = 0u;
    }
    return *this;
  }
  ~ScopedBuffer() { reset(); }

  void reset() {
    if (adapter != nullptr) {
      ReleaseVulkanBuffer(*adapter, buffer);
    }
    adapter = nullptr;
    used_bytes = 0u;
  }
};

#endif

}  // namespace rund::node::accel::detail
