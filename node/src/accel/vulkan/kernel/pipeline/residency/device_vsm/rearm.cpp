#include "../../../../buffer/access.hpp"
#include "../../../../buffer/resident/lookup.hpp"

#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

DeviceVsmRearmResult
rearm(const std::shared_ptr<void> &lowering,
      const std::shared_ptr<const DeviceVsmProof> &proof) noexcept {
  DeviceVsmRearmResult out{};
  const auto fail = [&out](const char *const why) noexcept {
    out.check = {false, why};
    return out;
  };
  const std::shared_ptr<Owner> owner = owner_of(lowering);
  if (owner == nullptr || owner->magic != OwnerMagic || owner->proof != proof ||
      proof == nullptr || !device_vsm_proof_valid(*proof)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  std::lock_guard lock{owner->gate};
  if (owner->adapter == nullptr ||
      owner->adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return fail("compute_device_lost");
  }
  if (owner->in_flight) {
    return fail("compute_pipeline_busy");
  }
  if (!owner->submitted) {
    return fail("accel_kernel_pipeline_invalid");
  }
  const auto ready = [](const VulkanBuffer &buffer,
                        const std::uint64_t bytes) noexcept {
    return bytes != 0u && bytes <= std::numeric_limits<VkDeviceSize>::max() &&
           buffer.buffer != VK_NULL_HANDLE && buffer.mapped != nullptr &&
           buffer.bytes >= static_cast<VkDeviceSize>(bytes) &&
           buffer.capacity_bytes >= static_cast<VkDeviceSize>(bytes);
  };
  std::array<std::uint32_t, DeviceVsmResultWordCount> zero{};
  zero[DeviceVsmFailedPageWord] = DeviceVsmNoFailedPage;
  if (!ready(owner->result.buffer, sizeof(zero))) {
    return fail("accel_vulkan_buffer_unavailable");
  }
  const bool graph = proof->topology == DeviceVsmTopology::GraphResident;
  if (graph && (owner->pipeline == nullptr || !owner->graph_recorded ||
                owner->graph_secondary.buffer == VK_NULL_HANDLE ||
                owner->graph_descriptors == VK_NULL_HANDLE ||
                owner->graph_digest != proof->graph_resident.digest ||
                !device_vsm_graph_resident_type_valid(
                    proof->graph_resident.type) ||
                proof->geometry.element_bytes !=
                    proof->graph_resident.type.element_bytes ||
                owner->graph_pipeline != owner->pipeline->pipeline ||
                owner->graph_layout != owner->pipeline->pipeline_layout)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  const bool graph_map =
      graph && device_vsm_page_map_active(proof->graph_resident.page_map);
  const bool graph_pointwise_map =
      proof->topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(proof->graph_pointwise.page_map);
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = graph_map
                                           ? proof->graph_resident.page_map
                                           : proof->graph_pointwise.page_map;
    if (!device_vsm_page_map_valid(page_map, proof->geometry.page_count) ||
        owner->params.buffer.buffer == VK_NULL_HANDLE ||
        owner->params.buffer.mapped == nullptr ||
        owner->params.buffer.bytes != DeviceVsmPageMapBytes ||
        owner->params.buffer.capacity_bytes < DeviceVsmPageMapBytes ||
        owner->params.buffer.allocated_bytes < DeviceVsmPageMapBytes) {
      return fail("accel_vulkan_buffer_unavailable");
    }
    const auto *const words =
        static_cast<const std::uint32_t *>(owner->params.buffer.mapped);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      if (words[index] != page_map.words[index]) {
        return fail("accel_vulkan_buffer_unavailable");
      }
    }
  }
  const bool ring = proof->topology == DeviceVsmTopology::Pointwise ||
                    proof->topology == DeviceVsmTopology::GraphPointwise;
  const bool window = proof->topology == DeviceVsmTopology::Window;
  GraphResidentBuffers expected_graph{};
  GraphResidentBuffers fresh_graph{};
  std::array<VulkanResidentBufferResult, DeviceVsmResidentCapacity>
      fresh_endpoints{};
  std::array<VulkanResidentBufferResult, DeviceVsmResidentCapacity>
      fresh_residents{};
  if (graph &&
      (owner->graph.ready == 0u || owner->graph.owner_count == 0u ||
       owner->graph.owner_count != proof->graph_resident.owner_binding_count ||
       owner->graph.endpoint_count != proof->residents.count ||
       owner->graph.endpoint_count == 0u)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  if (graph) {
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        expected_graph[slot][bank] = owner->graph.owners[slot].buffers[bank];
      }
    }
    if (!preflight_graph_bindings(owner->device, proof->graph_resident,
                                  &expected_graph, fresh_graph)) {
      return fail("accel_vulkan_buffer_unavailable");
    }
    for (std::size_t endpoint = 0u; endpoint < owner->graph.endpoint_count;
         ++endpoint) {
      const auto &source = proof->residents.rows[endpoint];
      const auto &old = owner->graph.endpoints[endpoint];
      VulkanResidentBufferResult current = LookupVulkanResidentBuffer(
          owner->device, source.backing, source.handle);
      if (!current.check.ok || current.device_buffer == nullptr ||
          current.storage == nullptr ||
          !device_vsm_graph_resident_same_object(current.handle,
                                                 source.handle) ||
          !same_ref(current.ref, old.ref) ||
          !device_vsm_graph_resident_same_object(current.handle, old.handle) ||
          !device_vsm_graph_resident_same_object(current.storage,
                                                 old.storage) ||
          current.device_buffer != old.device_buffer) {
        return fail("accel_vulkan_buffer_unavailable");
      }
      fresh_endpoints[endpoint] = std::move(current);
    }
  }
  if (!graph) {
    if (owner->resident_count != proof->residents.count ||
        owner->resident_count > DeviceVsmResidentCapacity) {
      return fail("accel_kernel_pipeline_invalid");
    }
    for (std::size_t index = 0u; index < owner->resident_count; ++index) {
      const auto &source = proof->residents.rows[index];
      const auto &old = owner->residents[index];
      VulkanResidentBufferResult current = LookupVulkanResidentBuffer(
          owner->device, source.backing, source.handle);
      if (!current.check.ok || current.device_buffer == nullptr ||
          current.storage == nullptr || !same_ref(current.ref, old.ref) ||
          !device_vsm_graph_resident_same_object(current.handle,
                                                 source.handle) ||
          !device_vsm_graph_resident_same_object(current.handle, old.handle) ||
          !device_vsm_graph_resident_same_object(current.storage,
                                                 old.storage) ||
          current.device_buffer != old.device_buffer) {
        return fail("accel_vulkan_buffer_unavailable");
      }
      fresh_residents[index] = std::move(current);
    }
    for (std::size_t index = owner->resident_count;
         index < DeviceVsmResidentCapacity; ++index) {
      const auto &source = proof->residents.rows[index];
      const auto &old = owner->residents[index];
      if (source.backing.id != 0u || source.handle != nullptr ||
          old.ref.id != 0u || old.handle != nullptr || old.storage != nullptr ||
          old.device_buffer != nullptr || old.check.ok ||
          old.storage_bytes != 0u || old.storage_reused) {
        return fail("accel_kernel_pipeline_invalid");
      }
    }
  }
  DeviceVsmWindowRingPlan window_ring_plan{};
  const bool window_ring =
      window && proof->window.ring.gpu_owned &&
      device_vsm_window_ring_plan_expected(proof->geometry, window_ring_plan) &&
      proof->window.ring == window_ring_plan;
  if (window && proof->window.ring.gpu_owned && !window_ring) {
    return fail("accel_vulkan_buffer_unavailable");
  }
  std::uint64_t state_bytes = 0u;
  std::uint64_t scratch_bytes = 0u;
  std::uint64_t region_bytes = 0u;
  if (ring) {
    std::uint64_t expected_scratch_bytes = 0u;
    if (!device_vsm_ring_storage_expected(proof->geometry, proof->width,
                                          proof->residents.count, state_bytes,
                                          scratch_bytes) ||
        !::rund::kernel::checked::mul(
            proof->width, proof->geometry.payload_bytes, region_bytes) ||
        !::rund::kernel::checked::mul(region_bytes, proof->residents.count,
                                     expected_scratch_bytes) ||
        scratch_bytes != expected_scratch_bytes ||
        owner->ring_scratch_count != proof->residents.count) {
      return fail("accel_vulkan_buffer_unavailable");
    }
  } else if (window_ring) {
    if (window_ring_plan.scratch_bytes == 0u ||
        window_ring_plan.scratch_bytes % DeviceVsmWindowRingSlotCount != 0u ||
        owner->ring_scratch_count != DeviceVsmWindowRingSlotCount) {
      return fail("accel_vulkan_buffer_unavailable");
    }
    state_bytes = window_ring_plan.state_bytes;
    region_bytes =
        window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount;
    scratch_bytes = window_ring_plan.scratch_bytes;
  }
  const std::size_t scratch_count =
      ring || window_ring ? static_cast<std::size_t>(owner->ring_scratch_count)
                          : 0u;
  if ((ring || window_ring) &&
      (scratch_count == 0u || scratch_count > owner->ring_scratch.size() ||
       !ready(owner->ring_state.buffer, state_bytes) ||
       !ready(owner->ring_scratch[0u].buffer, region_bytes) ||
       (scratch_count > 1u &&
        !ready(owner->ring_scratch[1u].buffer, region_bytes)))) {
    return fail("accel_vulkan_buffer_unavailable");
  }
  if (ring || window_ring) {
    for (std::size_t index = 2u; index < scratch_count; ++index) {
      if (!ready(owner->ring_scratch[index].buffer, region_bytes)) {
        return fail("accel_vulkan_buffer_unavailable");
      }
    }
  }
  out.mutated = true;
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = graph_map
                                           ? proof->graph_resident.page_map
                                           : proof->graph_pointwise.page_map;
    auto *const words =
        static_cast<std::uint32_t *>(owner->params.buffer.mapped);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      words[index] = page_map.words[index];
    }
  }
  if (!graph) {
    for (std::size_t index = 0u; index < owner->resident_count; ++index) {
      owner->residents[index] = std::move(fresh_residents[index]);
    }
  }
  std::memcpy(owner->result.buffer.mapped, zero.data(), sizeof(zero));
  if (ring || window_ring) {
    if (!ClearVulkanBuffer(owner->ring_state.buffer,
                           static_cast<VkDeviceSize>(state_bytes))) {
      return fail("accel_vulkan_buffer_unavailable");
    }
    for (std::size_t index = 0u; index < scratch_count; ++index) {
      if (!ClearVulkanBuffer(owner->ring_scratch[index].buffer,
                             static_cast<VkDeviceSize>(region_bytes))) {
        return fail("accel_vulkan_buffer_unavailable");
      }
    }
  }
  if (graph) {
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        owner->graph.owners[slot].buffers[bank] = fresh_graph[slot][bank];
      }
    }
    for (std::size_t endpoint = 0u; endpoint < owner->graph.endpoint_count;
         ++endpoint) {
      owner->graph.endpoints[endpoint] = fresh_endpoints[endpoint];
    }
  }
  owner->submitted = false;
  out.check = {true, "ok"};
  return out;
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
