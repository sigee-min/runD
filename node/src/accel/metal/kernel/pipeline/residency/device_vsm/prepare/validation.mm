#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail::metal_device_vsm::prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck ValidateShapeAndBindings(Owner &owner,
                                          const rund::AccelDevice &pick,
                                          const DeviceVsmProof &proof,
                                          Shape &shape) noexcept {
  shape = {};
  shape.graph = proof.topology == DeviceVsmTopology::GraphResident;
  shape.graph_map =
      shape.graph && device_vsm_page_map_active(proof.graph_resident.page_map);
  shape.graph_pointwise_map =
      proof.topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(proof.graph_pointwise.page_map);
  if (shape.graph_map &&
      !device_vsm_page_map_valid(proof.graph_resident.page_map,
                                 proof.geometry.page_count)) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  if (shape.graph_pointwise_map &&
      !device_vsm_page_map_valid(proof.graph_pointwise.page_map,
                                 proof.geometry.page_count)) {
    return {false, "device_vsm_graph_binding_invalid"};
  }
  shape.window_ring = proof.topology == DeviceVsmTopology::Window &&
                      proof.window.ring.gpu_owned &&
                      device_vsm_window_ring_plan_expected(
                          proof.geometry, shape.window_ring_plan) &&
                      proof.window.ring == shape.window_ring_plan;
  if (proof.topology == DeviceVsmTopology::Window &&
      proof.window.ring.gpu_owned && !shape.window_ring) {
    return {false, "device_vsm_ring_storage_invalid"};
  }
  if (shape.window_ring) {
    std::uint64_t config_bytes = 0u;
    if (shape.window_ring_plan.config_stride !=
            DeviceVsmWindowRingConfigStride ||
        sizeof(WindowRingConfig) > shape.window_ring_plan.config_stride ||
        !::rund::kernel::checked::mul(1u, shape.window_ring_plan.config_stride,
                                      config_bytes) ||
        config_bytes != shape.window_ring_plan.config_bytes ||
        config_bytes > std::numeric_limits<NSUInteger>::max() ||
        shape.window_ring_plan.page_count != proof.geometry.page_count ||
        shape.window_ring_plan.payload_elements !=
            proof.geometry.payload_bytes / proof.geometry.element_bytes ||
        shape.window_ring_plan.frame_elements !=
            proof.geometry.frame_bytes / proof.geometry.element_bytes ||
        shape.window_ring_plan.halo_elements !=
            proof.geometry.read_prefix_bytes / proof.geometry.element_bytes) {
      return {false, "compute_pipeline_capacity"};
    }
  }
  shape.ring = proof.topology == DeviceVsmTopology::Pointwise ||
               proof.topology == DeviceVsmTopology::GraphPointwise;
  shape.ring_storage = shape.ring || shape.window_ring;

  owner.resident_count = shape.graph ? 0u : proof.residents.count;
  for (std::size_t index = 0u; index < owner.resident_count; ++index) {
    const DeviceVsmResidentBinding &row = proof.residents.rows[index];
    owner.residents[index] =
        LookupMetalResidentBuffer(pick, row.backing, row.handle);
  }
  if (shape.graph) {
    if (!device_vsm_graph_resident_type_valid(proof.graph_resident.type) ||
        proof.geometry.element_bytes !=
            proof.graph_resident.type.element_bytes) {
      return {false, "device_vsm_graph_binding_invalid"};
    }
    owner.graph.owner_count = proof.graph_resident.owner_binding_count;
    owner.graph.endpoint_count = proof.residents.count;
    if (owner.graph.owner_count == 0u ||
        owner.graph.owner_count > DeviceVsmGraphResidentPhysicalCapacity ||
        owner.graph.endpoint_count == 0u ||
        owner.graph.endpoint_count > DeviceVsmResidentCapacity) {
      return {false, "device_vsm_graph_binding_invalid"};
    }
    for (std::size_t endpoint = 0u; endpoint < owner.graph.endpoint_count;
         ++endpoint) {
      const auto &row = proof.residents.rows[endpoint];
      owner.graph.endpoints[endpoint] =
          LookupMetalResidentBuffer(pick, row.backing, row.handle);
    }
    for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
      const auto &source = proof.graph_resident.owners[slot];
      if (source.valid == 0u ||
          source.bank_count != DeviceVsmGraphResidentBankCapacity) {
        return {false, "device_vsm_graph_binding_invalid"};
      }
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        owner.graph.owners[slot][bank] = LookupMetalResidentBuffer(
            pick, source.refs[bank], source.handles[bank]);
      }
    }
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_device_vsm::prepare
