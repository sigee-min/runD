#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_resident {

bool validate_lifetimes(const DeviceVsmGraphResidentProof &proof,
                        const ValidationContext &) noexcept {
  for (std::size_t resource_index = 0u;
       resource_index < proof.resource_count; ++resource_index) {
    const auto &resource = proof.resources[resource_index];
    std::uint32_t first = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t last = 0u;
    std::uint32_t writer = std::numeric_limits<std::uint32_t>::max();
    std::size_t writers = 0u;
    bool first_read = false;
    for (std::size_t stage = 0u; stage < proof.stage_count; ++stage) {
      for (std::size_t port = 0u; port < proof.stages[stage].port_count;
           ++port) {
        const auto &row = proof.stages[stage].ports[port];
        if (row.resource != resource.resource) {
          continue;
        }
        if (first == std::numeric_limits<std::uint32_t>::max()) {
          first = static_cast<std::uint32_t>(stage);
          first_read = row.access == 0u;
        }
        last = static_cast<std::uint32_t>(stage);
        if (row.access == 1u) {
          writer = static_cast<std::uint32_t>(stage);
          ++writers;
        }
      }
    }
    if (first != resource.first || last != resource.last) {
      return false;
    }
    if (resource.role == 0u) {
      if (resource.producer != std::numeric_limits<std::uint32_t>::max() ||
          writers != 0u || !first_read) {
        return false;
      }
    } else if (resource.role == 1u) {
      if (resource.producer == std::numeric_limits<std::uint32_t>::max() ||
          writers != 1u || writer != resource.producer ||
          first != resource.producer || first_read ||
          resource.last <= resource.producer) {
        return false;
      }
    } else if (resource.role == 2u) {
      if (resource.producer != std::numeric_limits<std::uint32_t>::max() ||
          writers != 1u || writer != resource.first || first_read ||
          resource.first != resource.last ||
          resource.last != proof.stage_count - 1u) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
