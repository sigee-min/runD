#include "../../../../adapter/error.hpp"

#include "internal.hpp"

#include "../../../../command/dispatch.hpp"
#include "../../../../command.hpp"
#include "../../../../runtime/timestamp.hpp"

#include <cstring>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

void accept_submission(Owner &owner) noexcept {
  if (owner.pending_request.submission_control != nullptr) {
    (void)owner.pending_request.submission_control->accept();
  }
}

[[nodiscard]] DeviceVsmNativeExecution
execute_graph(Owner &owner, KernelCompletion completion,
              void *const user) noexcept {
  if (owner.adapter == nullptr || owner.pipeline == nullptr ||
      !owner.graph_recorded || owner.graph_secondary.buffer == VK_NULL_HANDLE ||
      owner.graph_descriptors == VK_NULL_HANDLE ||
      owner.graph_pipeline != owner.pipeline->pipeline ||
      owner.graph_layout != owner.pipeline->pipeline_layout) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  VulkanAdapter &adapter = *owner.adapter;
  if (!BeginVulkanCommand(adapter)) {
    return {.accepted = {false, VulkanLastError(&adapter)}};
  }
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  vkCmdExecuteCommands(adapter.command_buffer, 1u,
                       &owner.graph_secondary.buffer);
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(
      adapter.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
  owner.pending_native = DeviceVsmNativeExecution{
      .accepted = {true, "ok"},
      .encoded_tile_count = DeviceVsmGraphPhysicalDispatchCount,
  };
  EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  owner.submit_begin_ns = MonotonicNanoseconds();
  if (!SubmitVulkanCommand(adapter, completion, user, true)) {
    return {.accepted = {false, VulkanLastError(&adapter)}};
  }
  accept_submission(owner);
  return owner.pending_native;
}

} // namespace

DeviceVsmNativeExecution execute(Owner &owner, KernelCompletion completion,
                                 void *const user) noexcept {
  if (owner.adapter == nullptr || owner.pipeline == nullptr ||
      owner.proof == nullptr) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  VulkanAdapter &adapter = *owner.adapter;
  BeginVulkanCollectiveDescriptorEpoch(adapter);
  VkDescriptorSet descriptors = VK_NULL_HANDLE;
  const bool graph = owner.proof->topology == DeviceVsmTopology::GraphResident;
  const bool graph_map =
      graph && device_vsm_page_map_active(owner.proof->graph_resident.page_map);
  const bool graph_pointwise_map =
      owner.proof->topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(owner.proof->graph_pointwise.page_map);
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map =
        graph_map ? owner.proof->graph_resident.page_map
                  : owner.proof->graph_pointwise.page_map;
    if (!device_vsm_page_map_valid(page_map,
                                   owner.proof->geometry.page_count) ||
        owner.params.buffer.buffer == VK_NULL_HANDLE ||
        owner.params.buffer.mapped == nullptr ||
        owner.params.buffer.bytes != DeviceVsmPageMapBytes ||
        owner.params.buffer.capacity_bytes < DeviceVsmPageMapBytes ||
        owner.params.buffer.allocated_bytes < DeviceVsmPageMapBytes) {
      return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
    }
    const auto *const words =
        static_cast<const std::uint32_t *>(owner.params.buffer.mapped);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      if (words[index] != page_map.words[index]) {
        return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
      }
    }
  }
  if (graph) {
    return execute_graph(owner, completion, user);
  }
  const bool ring = owner.proof->topology == DeviceVsmTopology::Pointwise ||
                    owner.proof->topology == DeviceVsmTopology::GraphPointwise;
  const bool window_ring = owner.proof->topology == DeviceVsmTopology::Window &&
                           owner.proof->window.ring.gpu_owned;
  if (ring && owner.ring_scratch_count != owner.resident_count) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  if (window_ring && owner.ring_scratch_count != DeviceVsmWindowRingSlotCount) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  std::array<VulkanStorageBinding, 3u * DeviceVsmResidentCapacity +
                                       DeviceVsmGraphResidentPhysicalCapacity +
                                       3u>
      bindings{};
  bindings[0u] = VulkanStorageBindingFor(owner.params.buffer);
  if (graph) {
    bindings[1u] = VulkanStorageBindingFor(owner.graph_table.buffer);
    bindings[2u] = VulkanStorageBindingFor(owner.result.buffer);
    std::size_t binding = 3u;
    if (owner.graph.owner_count == 0u || owner.graph.ready == 0u ||
        owner.graph_table.buffer.buffer == VK_NULL_HANDLE) {
      return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
    }
    for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
      if (owner.graph.owners[slot].valid == 0u) {
        return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
      }
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        const auto &buffer = owner.graph.owners[slot].buffers[bank];
        bindings[binding++] =
            VulkanStorageBindingFor(buffer.device_buffer, buffer.ref);
      }
    }
    if (owner.graph.endpoint_count == 0u ||
        owner.graph.endpoint_count > DeviceVsmResidentCapacity) {
      return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
    }
    for (std::size_t endpoint = 0u; endpoint < owner.graph.endpoint_count;
         ++endpoint) {
      const auto &buffer = owner.graph.endpoints[endpoint];
      if (!buffer.check.ok || buffer.device_buffer == nullptr) {
        return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
      }
      bindings[binding++] =
          VulkanStorageBindingFor(buffer.device_buffer, buffer.ref);
    }
  }
  for (std::size_t index = 0u; !graph && index < owner.resident_count;
       ++index) {
    bindings[index + 1u] =
        ring ? VulkanStorageBindingFor(owner.ring_scratch[index].buffer)
             : VulkanStorageBindingFor(
                   owner.residents[index].device_buffer,
                   owner.proof->residents.rows[index].backing);
  }
  if (!graph) {
    bindings[owner.resident_count + 1u] =
        VulkanStorageBindingFor(owner.result.buffer);
  }
  std::uint32_t descriptor_count = 0u;
  if (!descriptor_count_for(graph, window_ring, ring, owner.graph.owner_count,
                            owner.graph.endpoint_count, owner.resident_count,
                            descriptor_count)) {
    return {.accepted = {false, DescriptorCountOverflowReason}};
  }
  if (ring) {
    const std::size_t inputs = owner.proof->residents.input_count;
    const std::size_t output = owner.proof->residents.input_count;
    bindings[owner.resident_count + 2u] =
        VulkanStorageBindingFor(owner.ring_state.buffer);
    bindings[owner.resident_count + 3u] =
        VulkanStorageBindingFor(owner.ring_scratch[output].buffer);
    for (std::size_t input = 0u; input < inputs; ++input) {
      bindings[owner.resident_count + 4u + input] =
          VulkanStorageBindingFor(owner.ring_scratch[input].buffer);
      bindings[owner.resident_count + 4u + inputs + input] =
          VulkanStorageBindingFor(owner.residents[input].device_buffer,
                                  owner.proof->residents.rows[input].backing);
    }
    bindings[owner.resident_count + 4u + 2u * inputs] =
        VulkanStorageBindingFor(owner.residents[output].device_buffer,
                                owner.proof->residents.rows[output].backing);
  }
  if (window_ring) {
    bindings[4u] = VulkanStorageBindingFor(owner.ring_state.buffer);
    bindings[5u] = VulkanStorageBindingFor(owner.ring_scratch[0u].buffer);
    bindings[6u] = VulkanStorageBindingFor(owner.ring_scratch[1u].buffer);
  }
  descriptors = owner.descriptor_set;
  if (descriptors == VK_NULL_HANDLE) {
    try {
      owner.descriptor_leases.reserve(owner.descriptor_leases.size() + 1u);
    } catch (...) {
      return {.accepted = {false, "compute_pipeline_capacity"}};
    }
    VulkanLeaseScope leases{adapter, owner.descriptor_leases};
    if (!AcquireVulkanCollectiveDescriptorSet(
            adapter, *owner.pipeline, descriptor_count, owner.descriptor_set) ||
        !WriteVulkanStorageDescriptorSet(adapter, owner.descriptor_set,
                                         bindings.data(), descriptor_count)) {
      return {.accepted = {false, VulkanLastError(&adapter)}};
    }
    descriptors = owner.descriptor_set;
  }
  if (!BeginVulkanCommand(adapter)) {
    return {.accepted = {false, VulkanLastError(&adapter)}};
  }
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  const DeviceVsmProof &proof = *owner.proof;
  const Dispatch dispatch{
      .logical_elements = static_cast<std::uint32_t>(
          proof.geometry.logical_bytes / proof.geometry.element_bytes),
      .payload_elements = static_cast<std::uint32_t>(
          proof.geometry.payload_bytes / proof.geometry.element_bytes),
      .page_count = static_cast<std::uint32_t>(proof.geometry.page_count),
      .width = proof.width,
      .element_words = static_cast<std::uint32_t>(proof.geometry.element_bytes /
                                                  sizeof(std::uint32_t)),
  };
  const VkCommandBuffer command = adapter.command_buffer;
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     owner.pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        owner.pipeline->pipeline_layout, 0u, 1u, &descriptors,
                        0u, nullptr);
  std::uint32_t window_seed_dispatches = 0u;
  std::uint32_t window_compute_dispatches = 0u;
  std::uint32_t window_internal_dispatches = 0u;
  std::uint32_t encoded_tile_count = 0u;
  if (window_ring && !device_vsm_window_internal_dispatch_count_native(
                         proof.geometry.page_count,
                         window_internal_dispatches)) {
    return {.accepted = {false, DeviceVsmDispatchCountOverflowReason}};
  }
  if (window_ring) {
    const WindowRingDispatch ring_dispatch{
        .logical_elements = static_cast<std::uint32_t>(
            proof.geometry.logical_bytes / proof.geometry.element_bytes),
        .payload_elements = static_cast<std::uint32_t>(
            proof.geometry.payload_bytes / proof.geometry.element_bytes),
        .page_count = static_cast<std::uint32_t>(proof.geometry.page_count),
        .width = proof.width,
        .element_words = 1u,
        .frame_elements = proof.window.ring.frame_elements,
        .halo_elements = proof.window.ring.halo_elements,
    };
    PushVulkanConstants(command, owner.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                        sizeof(ring_dispatch), &ring_dispatch);
    DispatchVulkan(command, 1u, 1u, 1u);
    window_seed_dispatches = ring_dispatch.page_count;
    window_compute_dispatches = ring_dispatch.page_count;
  } else if (graph) {
    DispatchVulkan(command, 1u, 1u, 1u);
    encoded_tile_count = DeviceVsmGraphPhysicalDispatchCount;
  } else {
    PushVulkanConstants(command, owner.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(dispatch),
                        &dispatch);
    DispatchVulkan(command,
                   proof.topology == DeviceVsmTopology::GraphPointwise ||
                           proof.topology ==
                               DeviceVsmTopology::GraphMapReduce ||
                           proof.topology == DeviceVsmTopology::Scan ||
                           proof.topology == DeviceVsmTopology::Reduce
                       ? 1u
                       : proof.width,
                   1u, 1u);
  }
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0u, 1u, &barrier, 0u,
                       nullptr, 0u, nullptr);

  owner.pending_native = DeviceVsmNativeExecution{
      .accepted = {true, "ok"},
      .window_seed_dispatches = window_seed_dispatches,
      .window_compute_dispatches = window_compute_dispatches,
      .window_internal_dispatches = window_internal_dispatches,
      .encoded_tile_count = encoded_tile_count,
  };
  EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  owner.submit_begin_ns = MonotonicNanoseconds();
  if (!SubmitVulkanCommand(adapter, completion, user, true)) {
    return {.accepted = {false, VulkanLastError(&adapter)}};
  }
  accept_submission(owner);
  return owner.pending_native;
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
