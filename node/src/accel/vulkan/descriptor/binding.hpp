#pragma once

#include "../adapter/buffer.hpp"

#include <limits>

namespace rund::kernel {
struct ResidentBufferRef;
}

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanStorageBinding {
  const VulkanBuffer *buffer = nullptr;
  VkDeviceSize offset = 0u;
  VkDeviceSize range = 0u;
};

struct StorageRange final {
  VkDeviceSize base{};
  VkDeviceSize bytes{};
  std::uint64_t offset{};
  std::uint64_t count{};
};

[[nodiscard]] inline VulkanStorageBinding
VulkanStorageBindingFor(const VulkanBuffer &buffer) noexcept {
  return VulkanStorageBinding{&buffer, buffer.offset, buffer.bytes};
}

[[nodiscard]] inline VulkanStorageBinding
VulkanStorageBindingFor(const VulkanBuffer &buffer,
                        const VkDeviceSize offset,
                        const VkDeviceSize range = 0u) noexcept {
  if (offset > buffer.bytes ||
      buffer.offset > std::numeric_limits<VkDeviceSize>::max() - offset) {
    return {};
  }
  const VkDeviceSize available = buffer.bytes - offset;
  const VkDeviceSize bytes = range == 0u ? available : range;
  if (bytes == 0u || bytes > available) {
    return {};
  }
  return VulkanStorageBinding{&buffer, buffer.offset + offset, bytes};
}

[[nodiscard]] VulkanStorageBinding
VulkanStorageBindingFor(const VulkanBuffer *const buffer,
                        const rund::kernel::ResidentBufferRef &ref) noexcept;

#endif

} // namespace rund::node::accel::detail
