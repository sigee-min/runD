#include "internal.hpp"

#include "../../../../../../accel/kernel/residency/device_vsm/graph_resident.hpp"

#include <cstddef>

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_resident_stages(GraphResidentDraft &draft) noexcept {
  namespace accel = node::accel::detail;
  for (std::size_t stage_index = 0u; stage_index < draft.stages.size();
       ++stage_index) {
    auto &row = draft.result.stages[stage_index];
    const residency::TiledGraphStage &stage = draft.stages[stage_index];
    if (stage.ports.empty() ||
        stage.ports.size() > accel::DeviceVsmGraphResidentPortCapacity) {
      remember_graph_resident_reason(GraphResidentStagePortsInvalid,
                                     draft.reason);
      return false;
    }
    row.node = stage.node;
    row.domain = static_cast<std::uint8_t>(stage.domain);
    row.port_count = static_cast<std::uint16_t>(stage.ports.size());
    row.valid = 1u;
    std::size_t port_index = 0u;
    for (const residency::TiledGraphPort port : stage.ports) {
      if (draft.ports >= accel::DeviceVsmGraphResidentPortCapacity ||
          draft.plan.resource(port.resource) == nullptr ||
          (port.next_stage != residency::NoGraphStage &&
           port.next_stage >= draft.stages.size()) ||
          (port.access != residency::Access::Read &&
           port.access != residency::Access::Write)) {
        remember_graph_resident_reason(GraphResidentStagePortsInvalid,
                                       draft.reason);
        return false;
      }
      auto &port_row = row.ports[port_index++];
      port_row.resource = port.resource;
      port_row.next_stage = port.next_stage;
      port_row.program_port = port.program_port;
      port_row.access = static_cast<std::uint8_t>(port.access);
      port_row.valid = 1u;
      ++draft.ports;
    }
  }
  draft.result.resource_count =
      static_cast<std::uint32_t>(draft.resources.size());
  draft.result.stage_count = static_cast<std::uint32_t>(draft.stages.size());
  draft.result.port_count = static_cast<std::uint32_t>(draft.ports);
  draft.result.owner_binding_count =
      static_cast<std::uint32_t>(draft.internal_count);
  draft.result.valid = 1u;
  draft.result.digest = accel::device_vsm_graph_resident_digest(draft.result);
  if (draft.result.resources[0u].page_bytes == 0u ||
      draft.result.resources[0u].page_bytes % draft.result.type.element_bytes !=
          0u) {
    remember_graph_resident_reason(GraphResidentProofValidationInvalid,
                                   draft.reason);
    return false;
  }
  if (!accel::device_vsm_graph_resident_valid(
          draft.result,
          draft.result.resources[0u].page_bytes /
              draft.result.type.element_bytes,
          static_cast<std::uint32_t>(draft.graph_input_resources.size()),
          draft.plan.page_count())) {
    remember_graph_resident_reason(GraphResidentProofValidationInvalid,
                                   draft.reason);
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
