#include "../publish.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../command/dispatch.hpp"
#include "../../command.hpp"

#include <limits>

namespace rund::node::accel::detail {

bool EncodeVulkanPipelinePublish(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind !=
        static_cast<std::uint32_t>(PreparedKernelPublicationKind::Terminal)) {
      continue;
    }
    if (route.descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u) {
      return false;
    }
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(route.params),
                        &route.params);
    DispatchVulkan(command, route.groups_x, route.groups_y, 1u);
  }
  VkMemoryBarrier visible{};
  visible.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  visible.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  visible.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0u, 1u, &visible, 0u, nullptr, 0u, nullptr);
  return true;
}

bool EncodeVulkanPipelineCanonicalize(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources,
    const std::uint32_t state) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind != static_cast<std::uint32_t>(
                                 PreparedKernelPublicationKind::Terminal) ||
        route.params.state != state) {
      continue;
    }
    if (route.canonical_descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u || route.params.final >= 3u) {
      return false;
    }
    VulkanPipelinePublishParams params = route.params;
    params.target_offset_words = params.source_offset_words[params.final];
    params.target_stride_words = params.source_stride_words[params.final];
    params.stop = std::numeric_limits<std::uint32_t>::max();
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.canonical_descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    ::vkCmdDispatch(command, route.groups_x, route.groups_y, 1u);
  }
  EncodeVulkanComputeToComputeBarrier(command);
  return true;
}

bool EncodeVulkanPipelineWindowPublish(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources, const std::uint32_t state,
    const std::uint32_t outer) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind !=
            static_cast<std::uint32_t>(PreparedKernelPublicationKind::Window) ||
        route.params.state != state) {
      continue;
    }
    if (route.descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u || route.params.tile == 0u) {
      return false;
    }
    VulkanPipelinePublishParams params = route.params;
    params.outer = outer;
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    DispatchVulkan(command, route.groups_x, route.groups_y, 1u);
  }
  EncodeVulkanComputeToComputeBarrier(command);
  return true;
}

} // namespace rund::node::accel::detail

#endif
