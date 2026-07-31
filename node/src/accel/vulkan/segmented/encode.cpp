#include <accel/check.hpp>

#include "local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct SegmentedDispatch {
  rund::kernel::u64 base_block = 0u;
};

void BindSegmentedPhase(const VkCommandBuffer command,
                        const VulkanCollectivePipeline &pipeline,
                        const VkDescriptorSet set) {
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     pipeline.pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline.pipeline_layout, 0u, 1u, &set, 0u, nullptr);
}

void DispatchSegmentedChunks(const VulkanSegmentedScanEncodeResources &scan,
                             VulkanCollectivePipeline &pipeline,
                             const VkCommandBuffer command) {
  const rund::kernel::u64 limit = scan.adapter->max_dispatch_groups;
  for (rund::kernel::u64 base = 0u; base < scan.plan.block_count;) {
    const SegmentedDispatch dispatch{base};
    const auto groups = static_cast<std::uint32_t>(
        std::min(limit, scan.plan.block_count - base));
    PushVulkanConstants(command, pipeline.pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(dispatch),
                        &dispatch);
    DispatchVulkan(command, groups, 1u, 1u);
    base += groups;
  }
}

} // namespace
#endif

rund::AccelCheck
EncodeVulkanSegmentedScan(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources,
                          void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const scan =
      static_cast<VulkanSegmentedScanEncodeResources *>(resources.get());
  const VkCommandBuffer command =
      reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (scan == nullptr || scan->adapter != &adapter || scan->block == nullptr ||
      scan->block_set == VK_NULL_HANDLE || command == VK_NULL_HANDLE ||
      scan->adapter->max_dispatch_groups == 0u || scan->dispatch_count == 0u ||
      scan->block->push_bytes != kSegmentedScanPushBytes ||
      (scan->plan.pass_count != 1u && scan->plan.pass_count != 2u)) {
    SetVulkanLastError(adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  BindSegmentedPhase(command, *scan->block, scan->block_set);
  DispatchSegmentedChunks(*scan, *scan->block, command);
  if (scan->plan.pass_count == 2u) {
    if (scan->prefix == nullptr || scan->offset == nullptr ||
        scan->prefix_set == VK_NULL_HANDLE ||
        scan->offset_set == VK_NULL_HANDLE || scan->prefix->push_bytes != 0u ||
        scan->offset->push_bytes != kSegmentedScanPushBytes) {
      SetVulkanLastError(adapter, "compute_segmented_scan_invalid");
      return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
    }
    EncodeVulkanComputeToComputeBarrier(command);
    BindSegmentedPhase(command, *scan->prefix, scan->prefix_set);
    DispatchVulkan(command, 1u, 1u, 1u);
    EncodeVulkanComputeToComputeBarrier(command);
    BindSegmentedPhase(command, *scan->offset, scan->offset_set);
    DispatchSegmentedChunks(*scan, *scan->offset, command);
  }
  const std::array<const VulkanBuffer *, 1u> outputs{scan->output};
  if (!FinishVulkanStatus(command, scan->status, outputs)) {
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
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
