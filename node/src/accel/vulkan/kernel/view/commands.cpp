#include "internal.hpp"

#include "../../adapter/error.hpp"
#include "../../barrier.hpp"
#include "../../collective/pipeline.hpp"
#include "../../command/dispatch.hpp"
#include "../../command.hpp"
#include "../../descriptor.hpp"

#include <array>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

rund::AccelCheck EncodeTransfers(const VulkanViewLowering &view,
                                 const VkCommandBuffer command,
                                 const bool inputs) {
  if (view.transfers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (command == VK_NULL_HANDLE || view.pipeline == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  bool encoded = false;
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     view.pipeline->pipeline);
  for (const VulkanViewTransfer &transfer : view.transfers) {
    if (transfer.input != inputs) {
      continue;
    }
    const std::uint64_t element_words =
        transfer.element_bytes / sizeof(std::uint32_t);
    const std::uint64_t stride_words =
        transfer.stride_bytes / sizeof(std::uint32_t);
    for (const ViewPage &page : transfer.pages) {
      if (page.descriptor == VK_NULL_HANDLE || !page.grid.valid()) {
        return rund::AccelCheck{false, "compute_resident_stride_invalid"};
      }
      BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            view.pipeline->pipeline_layout, 0u, 1u,
                            &page.descriptor, 0u, nullptr);
      const VulkanViewParams params{
          .count = page.count,
          .source_offset_words =
              transfer.input ? page.external_words : page.dense_words,
          .source_stride_words = transfer.input ? stride_words : element_words,
          .target_offset_words =
              transfer.input ? page.dense_words : page.external_words,
          .target_stride_words = transfer.input ? element_words : stride_words,
          .element_words = static_cast<std::uint32_t>(element_words),
      };
      vkCmdPushConstants(command, view.pipeline->pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                         &params);
      DispatchVulkan(command, page.grid.x, page.grid.y, 1u);
      encoded = true;
    }
  }
  const bool closes_input_lifetime = !inputs && view.has_input;
  if (encoded || closes_input_lifetime) {
    EncodeVulkanComputeToComputeBarrier(command);
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck
PrepareVulkanViewCommands(VulkanAdapter &adapter,
                          const std::shared_ptr<VulkanViewLowering> &view) {
  if (view == nullptr || view->transfers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  const bool borrowed_pipeline = view->pipeline != nullptr;
  if (!borrowed_pipeline) {
    view->pipeline = AcquireVulkanViewPipeline(adapter);
  }
  if (view->pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  const std::uint64_t descriptor_set_count = VulkanViewDispatchCount(view);
  if (!borrowed_pipeline &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *view->pipeline, kVulkanViewDescriptorCount,
          descriptor_set_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  for (VulkanViewTransfer &transfer : view->transfers) {
    const VulkanBuffer *const external = transfer.external.device_buffer;
    const VulkanBuffer *const dense = transfer.dense.device_buffer;
    if (external == nullptr || dense == nullptr) {
      return rund::AccelCheck{false, "accel_buffer_unavailable"};
    }
    const VulkanStorageBinding dense_binding =
        VulkanStorageBindingFor(dense, transfer.dense.ref);
    for (ViewPage &page : transfer.pages) {
      if (!AcquireVulkanCollectiveDescriptorSet(adapter, *view->pipeline,
                                                kVulkanViewDescriptorCount,
                                                page.descriptor)) {
        return rund::AccelCheck{false, VulkanLastError(&adapter)};
      }
      const VulkanStorageBinding external_binding{external, page.base_bytes,
                                                  page.span_bytes};
      const std::array<VulkanStorageBinding, kVulkanViewDescriptorCount>
          bindings{transfer.input ? external_binding : dense_binding,
                   transfer.input ? dense_binding : external_binding};
      if (!WriteVulkanStorageDescriptorSet(adapter, page.descriptor, bindings)) {
        return rund::AccelCheck{false, VulkanLastError(&adapter)};
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
EncodeVulkanViewInputs(const std::shared_ptr<VulkanViewLowering> &view,
                       const VkCommandBuffer command) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, command, true);
}

rund::AccelCheck
EncodeVulkanViewOutputs(const std::shared_ptr<VulkanViewLowering> &view,
                        const VkCommandBuffer command) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, command, false);
}

#endif

} // namespace rund::node::accel::detail
