#pragma once

#include "capture.hpp"
#include "timestamp.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void BindVulkanPipeline(VkCommandBuffer command, VkPipelineBindPoint point,
                        VkPipeline pipeline) noexcept;

void BindVulkanDescriptors(VkCommandBuffer command, VkPipelineBindPoint point,
                           VkPipelineLayout layout, std::uint32_t first_set,
                           std::uint32_t set_count,
                           const VkDescriptorSet *sets,
                           std::uint32_t dynamic_count,
                           const std::uint32_t *dynamic_offsets) noexcept;

void PushVulkanConstants(VkCommandBuffer command, VkPipelineLayout layout,
                         VkShaderStageFlags stages, std::uint32_t offset,
                         std::uint32_t size, const void *data) noexcept;

void DispatchVulkan(VkCommandBuffer command, std::uint32_t x, std::uint32_t y,
                    std::uint32_t z) noexcept;

void DispatchVulkanIndirect(VkCommandBuffer command, VkBuffer source,
                            VkDeviceSize offset) noexcept;

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
