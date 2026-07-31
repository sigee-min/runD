#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodeVulkanHistogram(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources,
                                       void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const histogram =
      static_cast<VulkanHistogramEncodeResources *>(resources.get());
  const VkCommandBuffer command =
      reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (histogram == nullptr || histogram->adapter != &adapter ||
      command == VK_NULL_HANDLE || histogram->clear_pipeline == nullptr ||
      histogram->count_pipeline == nullptr || histogram->bins == nullptr ||
      histogram->counts == nullptr) {
    SetVulkanLastError(adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  const std::uint32_t clear_groups = static_cast<std::uint32_t>(
      (histogram->plan.bin_count + kHistogramBlockSize - 1u) /
      kHistogramBlockSize);
  const std::uint32_t count_groups = static_cast<std::uint32_t>(
      (histogram->plan.element_count + kHistogramBlockSize - 1u) /
      kHistogramBlockSize);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     histogram->clear_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        histogram->clear_pipeline->pipeline_layout, 0u, 1u,
                        &histogram->clear_descriptor_set, 0u, nullptr);
  DispatchVulkan(command, clear_groups, 1u, 1u);
  std::array<VkBufferMemoryBarrier, 2u> mid{
      VulkanBufferBarrier(*histogram->counts, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
      VulkanBufferBarrier(histogram->status.device, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT |
                              VK_ACCESS_SHADER_WRITE_BIT),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(mid.size()), mid.data(), 0u,
                       nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     histogram->count_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        histogram->count_pipeline->pipeline_layout, 0u, 1u,
                        &histogram->count_descriptor_set, 0u, nullptr);
  DispatchVulkan(command, count_groups, 1u, 1u);
  const std::array<const VulkanBuffer *, 1u> outputs{histogram->counts};
  if (!FinishVulkanStatus(command, histogram->status, outputs)) {
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
