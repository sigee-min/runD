#include "../../../../buffer/access.hpp"
#include "../../../../buffer/create.hpp"

#include "internal.hpp"

#include "../generated_indirect/internal.hpp"
#include "../generated_indirect/map.hpp"
#include "../mode.hpp"

#include "../../../../map/api.hpp"
#include "../../../../map/local.hpp"
#include "../../../local.hpp"
#include "../../prepare/record.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanResidencySlidingGate(
    VulkanPipeline &pipeline, VulkanResidencySelection &selection) noexcept {
  using namespace vulkan_sliding_detail;
  VulkanResidencySlidingGate &gate = selection.sliding;
  const auto access =
      vulkan_generated_indirect_detail::map_access(selection);
  if (!access.valid) {
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (access.selected) {
    return vulkan_generated_indirect_detail::prepare_gate(pipeline, selection);
  }
  if (selection.adapter != pipeline.adapter || !selection.bounded ||
      selection.original_arguments.empty() ||
      selection.original_arguments.size() != selection.argument_owners.size() ||
      selection.original_arguments.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      pipeline.control.generation_stride != 1u ||
      !coherent_storage(selection.arguments) ||
      !coherent_storage(pipeline.control.summary) ||
      !VulkanTimelineCapability(pipeline.adapter->timeline).ok ||
      pipeline.adapter->timeline.signal == nullptr) {
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  const VkDeviceSize argument_bytes =
      static_cast<VkDeviceSize>(selection.original_arguments.size()) *
      sizeof(VkDispatchIndirectCommand);
  const VkDeviceSize owner_bytes =
      static_cast<VkDeviceSize>(selection.argument_owners.size()) *
      sizeof(std::uint32_t);
  if (!CreateVulkanBuffer(
          *pipeline.adapter, sizeof(VulkanResidencySlidingPayload),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gate.descriptor) ||
      !CreateVulkanBuffer(*pipeline.adapter, argument_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          gate.original_arguments) ||
      !CreateVulkanBuffer(*pipeline.adapter, owner_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          gate.argument_owners) ||
      !coherent_storage(gate.descriptor) ||
      !coherent_storage(gate.original_arguments) ||
      !coherent_storage(gate.argument_owners) ||
      !UploadVulkanBuffer(gate.original_arguments,
                          selection.original_arguments.data(),
                          argument_bytes) ||
      !UploadVulkanBuffer(gate.argument_owners,
                          selection.argument_owners.data(), owner_bytes) ||
      !ClearVulkanBuffer(gate.descriptor, gate.descriptor.bytes)) {
    return {false, "accel_vulkan_memory_unavailable"};
  }
  const std::string_view source = gate_source();
  const auto artifact = VulkanFixedSourceArtifact(source);
  const auto plan = VulkanPipelineControlPlan(source);
  if (!artifact.ok) {
    return {false, artifact.reason};
  }
  try {
    gate.descriptor_leases.reserve(1u);
  } catch (...) {
    return {false, "compute_pipeline_capacity"};
  }
  {
    VulkanLeaseScope lease_scope{*pipeline.adapter, gate.descriptor_leases};
    gate.pipeline = AcquireVulkanCollectivePipeline(*pipeline.adapter, 5u, 0u,
                                                    plan, artifact);
    if (gate.pipeline == nullptr ||
        !ReserveVulkanCollectiveDescriptorDemand(*pipeline.adapter,
                                                 *gate.pipeline, 5u, 1u) ||
        !AcquireVulkanCollectiveDescriptorSet(*pipeline.adapter, *gate.pipeline,
                                              5u, gate.descriptor_set)) {
      return {false, "accel_vulkan_descriptor_unavailable"};
    }
    const std::array<const VulkanBuffer *, 5u> buffers{
        &gate.descriptor, &gate.original_arguments, &gate.argument_owners,
        &selection.arguments, &pipeline.control.summary};
    if (!WriteVulkanStorageDescriptorSet(*pipeline.adapter, gate.descriptor_set,
                                         buffers)) {
      return {false, "accel_vulkan_descriptor_unavailable"};
    }
  }
  if (!record_gate_command(pipeline, selection) ||
      !create_ready_semaphore(pipeline, gate)) {
    return {false, "accel_vulkan_command_unavailable"};
  }
  gate.retained_bytes = ::rund::detail::counter::SaturatingAdd(
      static_cast<std::uint64_t>(gate.descriptor.allocated_bytes),
      static_cast<std::uint64_t>(gate.original_arguments.allocated_bytes));
  gate.retained_bytes = ::rund::detail::counter::SaturatingAdd(
      gate.retained_bytes,
      static_cast<std::uint64_t>(gate.argument_owners.allocated_bytes));
  gate.retained_bytes = ::rund::detail::counter::SaturatingAdd(
      gate.retained_bytes,
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(gate.descriptor_leases.capacity()),
          sizeof(VulkanCollectiveDescriptorLease)));
  if (gate.retained_bytes == std::numeric_limits<std::uint64_t>::max()) {
    return {false, "compute_pipeline_capacity"};
  }
  gate.ready_for_submit = true;
  return {true, "ok"};
}

void DestroyVulkanResidencySlidingGate(
    VulkanResidencySelection &selection) noexcept {
  VulkanResidencySlidingGate &gate = selection.sliding;
  // Persistent owns the generated role command tables until terminal
  // acknowledgement.  Destruction while a run is live would leave the
  // tail/full-role aliases dangling and turn a later auth failure into an
  // unchecked use-after-destroy.
  if (selection.active_persistent.load(std::memory_order_acquire) != nullptr) {
    return;
  }
  if (selection.adapter != nullptr) {
    vulkan_generated_indirect_detail::destroy_roles(selection);
    if (gate.ready != VK_NULL_HANDLE) {
      vkDestroySemaphore(selection.adapter->device, gate.ready, nullptr);
    }
    DestroyCommand(selection.adapter->device, gate.command);
    ReleaseVulkanLeases(gate.descriptor_leases);
    DestroyVulkanBuffer(*selection.adapter, gate.argument_owners);
    DestroyVulkanBuffer(*selection.adapter, gate.original_arguments);
    DestroyVulkanBuffer(*selection.adapter, gate.descriptor);
  }
  gate.pipeline = nullptr;
  gate.descriptor_set = VK_NULL_HANDLE;
  gate.ready = VK_NULL_HANDLE;
  gate.next_ready_value = 1u;
  gate.owner = nullptr;
  gate.plan_identity = 0u;
  gate.token = 0u;
  gate.run_generation = 0u;
  gate.last_coordinate = 0u;
  gate.last_turn = 0u;
  gate.last_descriptor_generation = 0u;
  gate.stride = 0u;
  gate.slot = 0u;
  gate.retained_bytes = 0u;
  gate.published_row.fill(0u);
  gate.active_owner.reset();
  gate.force_stale_once.store(false, std::memory_order_relaxed);
  gate.pause_terminal_frontier_once.store(false, std::memory_order_relaxed);
  gate.terminal_frontier_release.store(true, std::memory_order_relaxed);
  gate.submit_frontier_count.store(0u, std::memory_order_relaxed);
  gate.terminal_frontier_count.store(0u, std::memory_order_relaxed);
  gate.gpu_result_read_count.store(0u, std::memory_order_relaxed);
  gate.authenticated = false;
  gate.ready_for_submit = false;
  vulkan_generated_indirect_detail::set_map_ready(selection, false);
  gate.quarantined.store(false, std::memory_order_release);
}

#endif

} // namespace rund::node::accel::detail
