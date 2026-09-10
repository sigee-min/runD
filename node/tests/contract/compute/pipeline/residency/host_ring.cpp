#include "local.hpp"

#include "src/compute/virtual/host_ring.hpp"

#include <cstdint>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckHostRingCapacities() {
  using rund::compute::detail::project_virtual_host_ring_capacities;
  using rund::compute::detail::VirtualHostRingCapacities;

  VirtualHostRingCapacities rings{};
  if (!project_virtual_host_ring_capacities(7u, 3u, 64u, 64u, 14u * 64u, 32u,
                                            rings) ||
      rings.input != 4u || rings.output != 3u ||
      rings.storage_bytes != 14u * 64u) {
    return 1;
  }
  if (!project_virtual_host_ring_capacities(7u, 3u, 64u, 64u, 12u * 64u, 32u,
                                            rings) ||
      rings.input != 3u || rings.output != 3u ||
      rings.storage_bytes != 12u * 64u) {
    return 2;
  }
  // A remainder too small for another input page may still extend the output
  // ring. Both decisions are deterministic and remain within the exact
  // two-bank budget.
  if (!project_virtual_host_ring_capacities(5u, 2u, 128u, 64u, 14u * 64u, 32u,
                                            rings) ||
      rings.input != 2u || rings.output != 3u ||
      rings.storage_bytes != 14u * 64u) {
    return 3;
  }
  if (project_virtual_host_ring_capacities(7u, 3u, 64u, 64u, 10u * 64u, 32u,
                                           rings) ||
      rings.input != 0u || rings.output != 0u || rings.storage_bytes != 0u) {
    return 4;
  }
  if (project_virtual_host_ring_capacities(
          7u, 3u, std::numeric_limits<std::uint64_t>::max(), 2u,
          std::numeric_limits<std::uint64_t>::max(), 32u, rings)) {
    return 5;
  }
  // Centered Window's default bank budget can reserve exactly two canonical
  // input neighbors without unnecessarily growing the output staging ring.
  if (!project_virtual_host_ring_capacities(9u, 2u, 64u, 64u, 12u * 64u, 32u,
                                            rings) ||
      rings.input != 4u || rings.output != 2u ||
      rings.storage_bytes != 12u * 64u) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
