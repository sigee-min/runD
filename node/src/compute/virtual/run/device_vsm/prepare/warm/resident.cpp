#include "local.hpp"

#include "../graph_wavefront.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {
namespace {

[[nodiscard]] bool
same_buffer_ref(const rund::kernel::ResidentBufferRef &left,
                const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

[[nodiscard]] bool same_resident_proof(
    const node::accel::detail::DeviceVsmGraphResidentProof &left,
    const node::accel::detail::DeviceVsmGraphResidentProof &right) noexcept {
  if (left.valid != right.valid || left.width != right.width ||
      left.type != right.type || left.digest != right.digest ||
      left.resource_count != right.resource_count ||
      left.stage_count != right.stage_count ||
      left.port_count != right.port_count ||
      left.owner_binding_count != right.owner_binding_count) {
    return false;
  }
  if (!node::accel::detail::device_vsm_page_map_equal(left.page_map,
                                                      right.page_map)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.resource_count; ++index) {
    if (left.resources[index].valid != right.resources[index].valid ||
        left.resources[index].resource != right.resources[index].resource ||
        left.resources[index].physical_id !=
            right.resources[index].physical_id ||
        left.resources[index].color != right.resources[index].color ||
        left.resources[index].producer != right.resources[index].producer ||
        left.resources[index].first != right.resources[index].first ||
        left.resources[index].last != right.resources[index].last ||
        left.resources[index].owner_slot != right.resources[index].owner_slot ||
        left.resources[index].external_slot !=
            right.resources[index].external_slot ||
        left.resources[index].page_bytes != right.resources[index].page_bytes ||
        left.resources[index].logical_bytes !=
            right.resources[index].logical_bytes ||
        left.resources[index].role != right.resources[index].role ||
        left.resources[index].type != right.resources[index].type ||
        left.resources[index].format != right.resources[index].format) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.owner_binding_count; ++index) {
    const auto &a = left.owners[index];
    const auto &b = right.owners[index];
    if (a.valid != b.valid || a.physical_id != b.physical_id ||
        a.color != b.color || a.view != b.view ||
        a.local_first != b.local_first || a.bank_count != b.bank_count ||
        a.physical_format != b.physical_format ||
        a.physical_page_bytes != b.physical_page_bytes ||
        a.physical_type != b.physical_type ||
        a.physical_tier != b.physical_tier ||
        a.physical_role != b.physical_role ||
        a.cache_regions != b.cache_regions ||
        a.bank_regions != b.bank_regions ||
        a.buffer_generations != b.buffer_generations) {
      return false;
    }
    for (std::size_t bank = 0u; bank < a.bank_count; ++bank) {
      if (!same_buffer_ref(a.refs[bank], b.refs[bank]) ||
          !node::accel::detail::device_vsm_graph_resident_same_object(
              a.handles[bank], b.handles[bank])) {
        return false;
      }
    }
  }
  for (std::size_t stage = 0u;
       stage < node::accel::detail::DeviceVsmGraphResidentStageCapacity;
       ++stage) {
    const auto &a = left.stages[stage];
    const auto &b = right.stages[stage];
    if (a.valid != b.valid || a.node != b.node || a.domain != b.domain ||
        a.port_count != b.port_count) {
      return false;
    }
    for (std::size_t port = 0u;
         port < node::accel::detail::DeviceVsmGraphResidentPortCapacity;
         ++port) {
      if (a.ports[port].valid != b.ports[port].valid ||
          a.ports[port].resource != b.ports[port].resource ||
          a.ports[port].next_stage != b.ports[port].next_stage ||
          a.ports[port].program_port != b.ports[port].program_port ||
          a.ports[port].access != b.ports[port].access) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

bool validate_resident(const VirtualPipelineState &state,
                       const VirtualRunProjection &run,
                       const DeviceVsmProductOwner &owner,
                       const char *&reason) noexcept {
  if (owner.pipeline_count < 2u || state.pipeline == nullptr ||
      state.pipeline->residency == nullptr ||
      state.pipeline->residency_pool == nullptr) {
    reason = "compute_device_vsm_graph_resident_shape_invalid";
    return false;
  }
  node::accel::detail::DeviceVsmGraphWavefrontProof fresh_wavefront{};
  if (!project_graph_wavefront(state.pipeline->residency->tiled_graph(),
                               page_count(run), owner.pipeline_stages[0u],
                               owner.pipeline_stages[owner.pipeline_count - 1u],
                               fresh_wavefront)) {
    reason = "compute_device_vsm_graph_resident_shape_invalid";
    return false;
  }
  node::accel::detail::DeviceVsmGraphResidentProof fresh{};
  const char *projection_reason = nullptr;
  if (!project_graph_resident(state.pipeline->residency->tiled_graph(),
                              *state.pipeline->residency_pool,
                              {state.graph_input_resources.data(),
                               state.graph_input_resource_count},
                              fresh_wavefront, fresh, projection_reason) ||
      !same_wavefront(fresh_wavefront, owner.proof->graph_wavefront)) {
    reason = projection_reason == nullptr
                 ? "compute_device_vsm_graph_resident_shape_invalid"
                 : projection_reason;
    return false;
  }
  if (!same_resident_proof(fresh, owner.proof->graph_resident)) {
    reason = "compute_device_vsm_graph_resident_shape_changed";
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail
