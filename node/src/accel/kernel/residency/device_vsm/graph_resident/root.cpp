#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_resident {

bool validate_root(const DeviceVsmGraphResidentProof &proof,
                   const std::uint64_t supplied_payload_elements,
                   const std::uint64_t page_count,
                   ValidationContext &context) noexcept {
  if (!(proof.valid != 0u && proof.width == DeviceVsmGraphResidentWidth &&
        device_vsm_graph_resident_type_valid(proof.type) &&
        proof.stage_count >= 2u &&
        proof.stage_count <= DeviceVsmGraphResidentStageCapacity &&
        proof.resource_count >= 2u &&
        proof.resource_count <= DeviceVsmGraphResidentResourceCapacity &&
        proof.port_count != 0u &&
        proof.port_count <= DeviceVsmGraphResidentPortCapacity &&
        proof.owner_binding_count != 0u &&
        proof.owner_binding_count <= DeviceVsmGraphResidentPhysicalCapacity &&
        proof.digest != 0u &&
        proof.digest == device_vsm_graph_resident_digest(proof))) {
    return false;
  }
  if (!device_vsm_page_map_valid(proof.page_map, page_count)) {
    return false;
  }
  std::uint64_t payload_elements = supplied_payload_elements;
  if (payload_elements == 0u) {
    if (proof.resources[0u].page_bytes == 0u ||
        proof.resources[0u].page_bytes % proof.type.element_bytes != 0u) {
      return false;
    }
    payload_elements = proof.resources[0u].page_bytes / proof.type.element_bytes;
  }
  if (payload_elements == 0u ||
      payload_elements > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  context.payload_elements = payload_elements;
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
