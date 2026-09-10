#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanDispatchCapture final {
  using Indirect = bool (*)(void *, VulkanDispatchCapture &, VkCommandBuffer,
                            VkBuffer, VkDeviceSize) noexcept;

  VkCommandBuffer command{VK_NULL_HANDLE};
  VkBuffer arguments{VK_NULL_HANDLE};
  VkDispatchIndirectCommand *mapped{};
  VkDispatchIndirectCommand *original{};
  std::uint32_t *owners{};
  std::size_t capacity{};
  std::size_t cursor{};
  std::uint32_t owner{};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  VkShaderStageFlags push_stages{};
  std::uint32_t push_offset{};
  std::uint32_t push_size{};
  std::array<std::byte, 256u> push{};
  void *context{};
  Indirect indirect{};
  std::uint64_t indirect_count{};
  bool has_push{};
  bool replay{};
  bool failed{};
};

extern thread_local VulkanDispatchCapture *vulkan_dispatch_capture;

class VulkanDispatchScope final {
public:
  VulkanDispatchScope(VulkanDispatchCapture &capture,
                      VkCommandBuffer command, std::uint32_t owner) noexcept
      : previous_{vulkan_dispatch_capture} {
    capture.command = command;
    capture.owner = owner;
    vulkan_dispatch_capture = &capture;
  }

  ~VulkanDispatchScope() { vulkan_dispatch_capture = previous_; }

  VulkanDispatchScope(const VulkanDispatchScope &) = delete;
  VulkanDispatchScope &operator=(const VulkanDispatchScope &) = delete;

private:
  VulkanDispatchCapture *previous_{};
};

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
