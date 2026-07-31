#pragma once

#include "../../adapter/buffer.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

struct VulkanResidentStorage final {
  VulkanAdapter *adapter = nullptr;
  VulkanBuffer buffer{};

  VulkanResidentStorage() = default;
  VulkanResidentStorage(const VulkanResidentStorage &) = delete;
  VulkanResidentStorage &operator=(const VulkanResidentStorage &) = delete;
  ~VulkanResidentStorage();
};

struct VulkanResidentOwner final {
  VulkanAdapter *adapter = nullptr;
  std::shared_ptr<void> adapter_owner{};
  std::shared_ptr<VulkanResidentStorage> storage{};
  std::uint64_t id = 0u;

  VulkanResidentOwner() = default;
  VulkanResidentOwner(const VulkanResidentOwner &) = delete;
  VulkanResidentOwner &operator=(const VulkanResidentOwner &) = delete;
  ~VulkanResidentOwner();
};

#endif

} // namespace rund::node::accel::detail
