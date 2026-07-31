#pragma once

#include "../cached/pipeline.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanMapDispatch final {
  std::uint32_t tile_count{};
  std::uint32_t iterations{1u};
};

static_assert(sizeof(VulkanMapDispatch) ==
              rund::kernel::compute_lowering_detail::kVulkanMapPushBytes);

inline void EncodeVulkanMap(VkCommandBuffer command,
                            const VulkanCachedPipeline &pipeline,
                            const VkDescriptorSet descriptors,
                            const std::uint32_t tile_count,
                            const std::uint32_t iterations) {
  const std::uint32_t groups =
      rund::kernel::compute_lowering_detail::VulkanMapGroupsForTiles(
          tile_count);
  const VulkanMapDispatch dispatch{
      .tile_count = tile_count,
      .iterations = iterations,
  };
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     pipeline.pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline.pipeline_layout, 0u, 1u, &descriptors, 0u,
                        nullptr);
  PushVulkanConstants(
      command, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
      rund::kernel::compute_lowering_detail::kVulkanMapPushBytes, &dispatch);
  DispatchVulkan(command, groups, 1u, 1u);
}

inline void EncodeVulkanMapIndirect(VkCommandBuffer command,
                                    const VulkanCachedPipeline &pipeline,
                                    const VkDescriptorSet descriptors,
                                    const std::uint32_t window_index,
                                    const std::uint32_t iterations,
                                    const VkBuffer indirect,
                                    const VkDeviceSize indirect_offset) {
  const VulkanMapDispatch dispatch{
      .tile_count = window_index,
      .iterations = iterations,
  };
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     pipeline.pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline.pipeline_layout, 0u, 1u, &descriptors, 0u,
                        nullptr);
  PushVulkanConstants(
      command, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
      rund::kernel::compute_lowering_detail::kVulkanMapPushBytes, &dispatch);
  DispatchVulkanIndirect(command, indirect, indirect_offset);
}

#endif

} // namespace rund::node::accel::detail
