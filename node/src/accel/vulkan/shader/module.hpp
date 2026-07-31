#pragma once

#include "../adapter/shader.hpp"

#include <accel/check.hpp>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

class VulkanModule final {
public:
  VulkanModule() noexcept = default;
  ~VulkanModule();
  VulkanModule(const VulkanModule &) = delete;
  VulkanModule &operator=(const VulkanModule &) = delete;
  VulkanModule(VulkanModule &&other) noexcept;
  VulkanModule &operator=(VulkanModule &&other) noexcept;

  [[nodiscard]] VkShaderModule get() const noexcept { return handle_; }
  void reset() noexcept;

private:
  friend rund::AccelCheck CreateVulkanModule(VulkanAdapter &,
                                             const VulkanShader &,
                                             VulkanModule &) noexcept;

  VulkanAdapter *adapter_{};
  VkShaderModule handle_{VK_NULL_HANDLE};
};

[[nodiscard]] rund::AccelCheck
CreateVulkanModule(VulkanAdapter &adapter, const VulkanShader &shader,
                   VulkanModule &module) noexcept;

#endif

} // namespace rund::node::accel::detail
