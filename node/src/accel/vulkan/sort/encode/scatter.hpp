#pragma once

#include <accel/check.hpp>

#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void EncodeVulkanSortScatter(const VulkanSortEncodeResources &sort,
                             const std::size_t pass,
                             const VkCommandBuffer command) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     sort.scatter_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sort.scatter_pipeline->pipeline_layout, 0u, 1u,
                        &sort.descriptors[pass].scatter_set, 0u, nullptr);
  DispatchVulkanSortChunks(sort, *sort.scatter_pipeline, command);
}

[[nodiscard]] rund::AccelCheck EncodeVulkanSortTargetBarrier(
    VulkanAdapter &adapter, const VulkanSortEncodeResources &sort,
    const std::size_t pass, const VkCommandBuffer command) {
  const VulkanBuffer *const target_keys = sort.target_keys[pass];
  const VulkanBuffer *const target_values = sort.target_values[pass];
  if (target_keys == nullptr || target_values == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_buffer_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_buffer_unavailable"};
  }
  const bool final_pass = pass + 1u == sort.pass_count;
  if (final_pass) {
    const std::array<const VulkanBuffer *, 2u> outputs{target_keys,
                                                       target_values};
    if (!FinishVulkanStatus(command, sort.status, outputs)) {
      SetVulkanLastError(adapter, "accel_vulkan_buffer_unavailable");
      return rund::AccelCheck{false, "accel_vulkan_buffer_unavailable"};
    }
    return rund::AccelCheck{true, "ok"};
  }
  constexpr VkAccessFlags target_dst =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  const std::array<VkBufferMemoryBarrier, 2u> barriers{
      VulkanBufferBarrier(*target_keys, VK_ACCESS_SHADER_WRITE_BIT, target_dst),
      VulkanBufferBarrier(*target_values, VK_ACCESS_SHADER_WRITE_BIT,
                          target_dst),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(barriers.size()),
                       barriers.data(), 0u, nullptr);
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
