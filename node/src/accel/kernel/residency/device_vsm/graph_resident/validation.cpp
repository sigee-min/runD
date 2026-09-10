#include "internal.hpp"

namespace rund::node::accel::detail {

bool device_vsm_graph_resident_valid(
    const DeviceVsmGraphResidentProof &proof,
    const std::uint64_t supplied_payload_elements,
    const std::uint32_t external_input_count,
    const std::uint64_t page_count) noexcept {
  device_vsm_graph_resident::ValidationContext context{};
  if (!device_vsm_graph_resident::validate_root(
          proof, supplied_payload_elements, page_count, context) ||
      !device_vsm_graph_resident::validate_owners(proof, context) ||
      !device_vsm_graph_resident::validate_resources(
          proof, external_input_count, context) ||
      !device_vsm_graph_resident::validate_owner_usage(proof, context) ||
      !device_vsm_graph_resident::validate_stages(proof, context) ||
      !device_vsm_graph_resident::validate_lifetimes(proof, context) ||
      !device_vsm_graph_resident::validate_tails(proof)) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
