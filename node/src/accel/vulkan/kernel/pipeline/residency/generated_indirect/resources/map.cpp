#include "../../../../../buffer/access.hpp"
#include "../../../../../buffer/create.hpp"

#include "../internal.hpp"
#include "../map.hpp"
#include "../map_state.hpp"

#include "../../../../../map/api.hpp"
#include "../../../../../map/admission.hpp"
#include "../../../../../map/encode/control.hpp"
#include "../../../../../map/source/control.hpp"
#include "../../../../../map/local.hpp"
#include "../../../prepare/record.hpp"
#include "../../sliding/internal.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

MapAccess map_access(const VulkanResidencySelection &selection) noexcept {
  const VulkanResidencyMode route = selection.mode;
  const bool selected = vulkan_residency_detail::is_map(route);
  const bool direct = vulkan_residency_detail::is_direct(route);
  const bool graph = vulkan_residency_detail::is_graph(route);
  if (!selected && !direct) {
    return MapAccess{.graph = graph, .recognized = graph};
  }
  if (direct) {
    return MapAccess{.valid = true,
                     .direct = true,
                     .recognized = true,
                     .layout_ready = true,
                     .resource_ready = true};
  }
  const bool active = selection.sliding.map_ready;
  const VulkanBuffer &descriptor = selection.sliding.descriptor;
  const bool resource_ready = descriptor.buffer != VK_NULL_HANDLE &&
                               descriptor.memory != VK_NULL_HANDLE &&
                               descriptor.mapped != nullptr;
  const bool layout_ready =
      descriptor.bytes >= sizeof(VulkanResidencySlidingPayload);
  return MapAccess{
      .valid = true,
      .selected = true,
      .recognized = true,
      .active = active,
      .descriptor_ready = active && layout_ready && resource_ready &&
                          vulkan_sliding_detail::coherent_storage(descriptor),
      .layout_ready = layout_ready,
      .resource_ready = resource_ready};
}

void set_map_ready(VulkanResidencySelection &selection,
                   const bool ready) noexcept {
  if (ready && !vulkan_residency_detail::is_map(selection.mode)) {
    return;
  }
  selection.sliding.map_ready = ready;
}

rund::AccelCheck prepare_gate(VulkanPipeline &pipeline,
                              VulkanResidencySelection &selection) noexcept {
  VulkanResidencySlidingGate &gate = selection.sliding;
  VulkanMapEncodeResources *checked_map = nullptr;
  if (pipeline.record == nullptr || !eligible(pipeline)) {
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  for (const VulkanPipelineRecordEntry &record_entry :
       pipeline.record->entries) {
    auto *const resources =
        static_cast<VulkanKernelResources *>(record_entry.prepared.get());
    const VulkanKernelEntry *const step =
        resources == nullptr ? nullptr : resources->entry(0u);
    auto *const map =
        step == nullptr
            ? nullptr
            : static_cast<VulkanMapEncodeResources *>(step->resource.get());
    if (map == nullptr ||
        (!map->prepared->checks.empty() && checked_map != nullptr)) {
      return {false, "accel_vulkan_pipeline_selection_unavailable"};
    }
    if (!map->prepared->checks.empty()) {
      checked_map = map;
    }
  }
  if (selection.adapter != pipeline.adapter || checked_map == nullptr ||
      !vulkan_sliding_detail::coherent_storage(pipeline.control.summary)) {
    return {false, "accel_vulkan_memory_unavailable"};
  }
  const rund::AccelCheck timeline =
      VulkanTimelineCapability(pipeline.adapter->timeline);
  if (!timeline.ok) {
    return timeline;
  }
  if (pipeline.adapter->timeline.signal == nullptr) {
    return {false, "accel_vulkan_timeline_failed"};
  }
  if (!CreateVulkanBuffer(
          *pipeline.adapter, sizeof(VulkanResidencySlidingPayload),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gate.descriptor) ||
      !vulkan_sliding_detail::coherent_storage(gate.descriptor) ||
      !ClearVulkanBuffer(gate.descriptor, gate.descriptor.bytes) ||
      gate.descriptor.bytes < sizeof(VulkanResidencySlidingPayload)) {
    return {false, "accel_vulkan_memory_unavailable"};
  }
  const auto artifact =
      VulkanGeneratedMapControlArtifact(checked_map->prepared->plan);
  if (!artifact.ok || artifact.source_text.empty()) {
    return {false, artifact.reason};
  }
  checked_map->generated_control_pipeline = AcquireVulkanCollectivePipeline(
      *pipeline.adapter, 6u, 128u, checked_map->prepared->plan, artifact);
  if (checked_map->generated_control_pipeline == nullptr ||
      !ReserveVulkanCollectiveDescriptorDemand(
          *pipeline.adapter, *checked_map->generated_control_pipeline, 6u,
          1u) ||
      !AcquireVulkanCollectiveDescriptorSet(
          *pipeline.adapter, *checked_map->generated_control_pipeline, 6u,
          checked_map->generated_control_descriptor)) {
    return {false, "accel_vulkan_descriptor_unavailable"};
  }
  bool bound = false;
  {
    MapModeGuard generated{*checked_map};
    bound = BindVulkanGeneratedMapAdmission(
        *checked_map, gate.descriptor, pipeline.control.summary);
  }
  if (!bound) {
    return {false, "accel_vulkan_descriptor_unavailable"};
  }
  set_map_ready(selection, true);
  gate.retained_bytes =
      static_cast<std::uint64_t>(gate.descriptor.allocated_bytes);
  gate.ready_for_submit = false;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
