#include "../../../../../buffer/resident/lookup.hpp"

#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_device_vsm::prepare {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck ValidateShape(const DeviceVsmProof &proof, bool &ring,
                               bool &window_ring, bool &ring_storage,
                               bool &graph, bool &graph_map,
                               bool &graph_pointwise_map,
                               DeviceVsmWindowRingPlan &window_ring_plan,
                               std::uint64_t &ring_state_bytes,
                               std::uint64_t &ring_scratch_bytes,
                               std::uint64_t &ring_region_bytes) noexcept {
  ring = proof.topology == DeviceVsmTopology::Pointwise ||
         proof.topology == DeviceVsmTopology::GraphPointwise;
  window_ring_plan = {};
  window_ring =
      proof.topology == DeviceVsmTopology::Window &&
      proof.window.ring.gpu_owned &&
      device_vsm_window_ring_plan_expected(proof.geometry, window_ring_plan) &&
      proof.window.ring == window_ring_plan;
  if (proof.topology == DeviceVsmTopology::Window &&
      proof.window.ring.gpu_owned && !window_ring) {
    return {false, "device_vsm_ring_storage_invalid"};
  }
  ring_storage = ring || window_ring;
  graph = proof.topology == DeviceVsmTopology::GraphResident;
  graph_map =
      graph && device_vsm_page_map_active(proof.graph_resident.page_map);
  graph_pointwise_map =
      proof.topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(proof.graph_pointwise.page_map);
  if (graph_map && !device_vsm_page_map_valid(proof.graph_resident.page_map,
                                              proof.geometry.page_count)) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  if (graph_pointwise_map &&
      !device_vsm_page_map_valid(proof.graph_pointwise.page_map,
                                 proof.geometry.page_count)) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  ring_state_bytes = 0u;
  ring_scratch_bytes = 0u;
  ring_region_bytes = 0u;
  std::uint64_t expected_ring_scratch_bytes = 0u;
  if (ring && !device_vsm_ring_storage_expected(
                  proof.geometry, proof.width, proof.residents.count,
                  ring_state_bytes, ring_scratch_bytes)) {
    return {false, "device_vsm_ring_storage_invalid"};
  }
  if (ring &&
      (!::rund::kernel::checked::mul(proof.width, proof.geometry.payload_bytes,
                                     ring_region_bytes) ||
       !::rund::kernel::checked::mul(ring_region_bytes, proof.residents.count,
                                     expected_ring_scratch_bytes))) {
    return {false, "device_vsm_ring_region_overflow"};
  }
  if (ring && ring_scratch_bytes != expected_ring_scratch_bytes) {
    return {false, "device_vsm_ring_scratch_mismatch"};
  }
  return {true, "ok"};
}

rund::AccelCheck BindResidents(const rund::AccelDevice &pick,
                               const DeviceVsmProof &proof,
                               Owner &owner) noexcept {
  const bool graph = proof.topology == DeviceVsmTopology::GraphResident;
  owner.resident_count = graph ? 0u : proof.residents.count;
  for (std::size_t index = 0u; index < owner.resident_count; ++index) {
    const DeviceVsmResidentBinding &row = proof.residents.rows[index];
    owner.residents[index] =
        LookupVulkanResidentBuffer(pick, row.backing, row.handle);
    if (!owner.residents[index].check.ok ||
        owner.residents[index].device_buffer == nullptr) {
      return {false, "device_vsm_resident_binding_invalid"};
    }
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm::prepare
