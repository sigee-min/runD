#include "../../../../../adapter/error.hpp"
#include "../../../../../buffer/resident/lookup.hpp"

#include "local.hpp"

#include "../../../../../command/dispatch.hpp"

#include <utility>

namespace rund::node::accel::detail::vulkan_device_vsm::prepare {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareGraph(const rund::AccelDevice &pick,
                              const DeviceVsmProof &proof,
                              Owner &owner) noexcept {
  if (!device_vsm_graph_resident_type_valid(proof.graph_resident.type) ||
      proof.geometry.element_bytes != proof.graph_resident.type.element_bytes) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  owner.graph.owner_count = proof.graph_resident.owner_binding_count;
  owner.graph.endpoint_count = proof.residents.count;
  if (owner.graph.endpoint_count == 0u ||
      owner.graph.endpoint_count > DeviceVsmResidentCapacity) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  for (std::size_t index = 0u; index < owner.graph.endpoint_count; ++index) {
    const auto &source = proof.residents.rows[index];
    VulkanResidentBufferResult current =
        LookupVulkanResidentBuffer(pick, source.backing, source.handle);
    if (!current.check.ok || current.device_buffer == nullptr ||
        current.storage == nullptr ||
        !device_vsm_graph_resident_same_object(current.handle, source.handle)) {
      return {false, "device_vsm_resident_binding_invalid"};
    }
    owner.graph.endpoints[index] = std::move(current);
  }
  GraphResidentBuffers graph_buffers{};
  if (!preflight_graph_bindings(pick, proof.graph_resident, nullptr,
                                graph_buffers)) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  GraphResidentProofTable table{};
  table.logical_elements = static_cast<std::uint32_t>(
      proof.geometry.logical_bytes / proof.geometry.element_bytes);
  table.payload_elements = static_cast<std::uint32_t>(
      proof.geometry.payload_bytes / proof.geometry.element_bytes);
  table.page_count = static_cast<std::uint32_t>(proof.geometry.page_count);
  table.element_words = static_cast<std::uint32_t>(
      proof.graph_resident.type.element_bytes / sizeof(std::uint32_t));
  table.element_bytes = proof.graph_resident.type.element_bytes;
  table.valid = 1u;
  table.owner_count = owner.graph.owner_count;
  table.controller = device_vsm_graph_controller(proof.graph_wavefront);
  for (std::size_t index = 0u; index < owner.graph.owner_count; ++index) {
    table.owner_valid[index] = 1u;
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      table.owner_generations[index * DeviceVsmGraphResidentBankCapacity +
                              bank] =
          proof.graph_resident.owners[index].buffer_generations[bank];
    }
  }
  if (owner.adapter == nullptr ||
      !MakeHostBuffer(*owner.adapter, &table, sizeof(table),
                      owner.graph_table)) {
    return {false, "device_vsm_graph_table_buffer_failed"};
  }
  for (std::size_t index = 0u; index < owner.graph.owner_count; ++index) {
    auto &target = owner.graph.owners[index];
    target.slot = static_cast<std::uint32_t>(index);
    target.valid = 1u;
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      target.buffers[bank] = graph_buffers[index][bank];
    }
  }
  owner.graph.ready = 1u;
  return {true, "ok"};
}

bool FillGraphBindings(Owner &owner) noexcept {
  if (owner.graph.owner_count == 0u || owner.graph.endpoint_count == 0u ||
      owner.graph.owner_count > DeviceVsmGraphResidentPhysicalCapacity ||
      owner.graph.endpoint_count > DeviceVsmResidentCapacity) {
    return false;
  }
  std::size_t binding = 0u;
  owner.graph_bindings[binding++] =
      VulkanStorageBindingFor(owner.params.buffer);
  owner.graph_bindings[binding++] =
      VulkanStorageBindingFor(owner.graph_table.buffer);
  owner.graph_bindings[binding++] =
      VulkanStorageBindingFor(owner.result.buffer);
  for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      const auto &row = owner.graph.owners[slot].buffers[bank];
      if (row.device_buffer == nullptr || !row.check.ok) {
        return false;
      }
      owner.graph_bindings[binding++] =
          VulkanStorageBindingFor(row.device_buffer, row.ref);
    }
  }
  for (std::size_t endpoint = 0u; endpoint < owner.graph.endpoint_count;
       ++endpoint) {
    const auto &row = owner.graph.endpoints[endpoint];
    if (row.device_buffer == nullptr || !row.check.ok) {
      return false;
    }
    owner.graph_bindings[binding++] =
        VulkanStorageBindingFor(row.device_buffer, row.ref);
  }
  if (binding > GraphBindingCapacity) {
    return false;
  }
  owner.graph_descriptor_count = static_cast<std::uint32_t>(binding);
  return true;
}

rund::AccelCheck RecordGraph(Owner &owner) noexcept {
  if (owner.adapter == nullptr || owner.pipeline == nullptr ||
      owner.graph_descriptor_count == 0u || !owner.graph.ready) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  try {
    owner.descriptor_leases.reserve(1u);
  } catch (...) {
    return {false, "compute_pipeline_capacity"};
  }
  VulkanAdapter &adapter = *owner.adapter;
  VulkanLeaseScope leases{adapter, owner.descriptor_leases};
  if (!AcquireVulkanCollectiveDescriptorSet(adapter, *owner.pipeline,
                                            owner.graph_descriptor_count,
                                            owner.graph_descriptors) ||
      !WriteVulkanStorageDescriptorSet(adapter, owner.graph_descriptors,
                                       owner.graph_bindings.data(),
                                       owner.graph_descriptor_count)) {
    return {false, VulkanLastError(&adapter)};
  }
  const rund::AccelCheck created =
      CreateCommand(adapter.device, adapter.compute_queue_family,
                    owner.graph_secondary, CommandKind::ImmutableSecondary);
  if (!created.ok) {
    return created;
  }
  const rund::AccelCheck begun = BeginCommand(
      adapter.device, owner.graph_secondary, CommandKind::ImmutableSecondary);
  if (!begun.ok) {
    return begun;
  }
  BindVulkanPipeline(owner.graph_secondary.buffer,
                     VK_PIPELINE_BIND_POINT_COMPUTE, owner.pipeline->pipeline);
  BindVulkanDescriptors(owner.graph_secondary.buffer,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        owner.pipeline->pipeline_layout, 0u, 1u,
                        &owner.graph_descriptors, 0u, nullptr);
  DispatchVulkan(owner.graph_secondary.buffer, 1u, 1u, 1u);
  const rund::AccelCheck ended = EndCommand(owner.graph_secondary);
  if (!ended.ok) {
    return ended;
  }
  owner.graph_pipeline = owner.pipeline->pipeline;
  owner.graph_layout = owner.pipeline->pipeline_layout;
  owner.graph_digest = owner.proof->graph_resident.digest;
  owner.graph_recorded = true;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm::prepare
