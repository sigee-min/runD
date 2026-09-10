#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_resident {

bool validate_stages(const DeviceVsmGraphResidentProof &proof,
                     ValidationContext &context) noexcept {
  context.port_total = 0u;
  for (std::size_t index = 0u; index < proof.stage_count; ++index) {
    const auto &stage = proof.stages[index];
    if (stage.valid == 0u || stage.domain > 2u || stage.port_count == 0u ||
        stage.port_count > DeviceVsmGraphResidentPortCapacity) {
      return false;
    }
    context.port_total += stage.port_count;
    if (context.port_total > DeviceVsmGraphResidentPortCapacity) {
      return false;
    }
    std::array<bool, DeviceVsmGraphResidentPortCapacity> read_ports{};
    std::array<bool, DeviceVsmGraphResidentPortCapacity> write_ports{};
    std::size_t reads = 0u;
    std::size_t writes = 0u;
    for (std::size_t port = stage.port_count;
         port < DeviceVsmGraphResidentPortCapacity; ++port) {
      const auto &row = stage.ports[port];
      if (row.valid != 0u || row.resource != 0u || row.next_stage != 0u ||
          row.program_port != 0u || row.access != 0u) {
        return false;
      }
    }
    for (std::size_t port = 0u; port < stage.port_count; ++port) {
      const auto &row = stage.ports[port];
      if (row.valid == 0u || row.resource == 0u || row.access > 1u ||
          (row.next_stage != std::numeric_limits<std::uint32_t>::max() &&
           row.next_stage >= proof.stage_count)) {
        return false;
      }
      if (row.access == 0u) {
        ++reads;
        if (row.program_port >= DeviceVsmGraphResidentPortCapacity ||
            read_ports[row.program_port]) {
          return false;
        }
        read_ports[row.program_port] = true;
      } else {
        ++writes;
        if (row.program_port >= DeviceVsmGraphResidentPortCapacity ||
            write_ports[row.program_port]) {
          return false;
        }
        write_ports[row.program_port] = true;
      }
      bool found = false;
      for (std::size_t resource = 0u; resource < proof.resource_count;
           ++resource) {
        found = found || proof.resources[resource].resource == row.resource;
      }
      if (!found) {
        return false;
      }
      if (row.next_stage != std::numeric_limits<std::uint32_t>::max()) {
        bool linked = false;
        const auto &target = proof.stages[row.next_stage];
        for (std::size_t target_port = 0u; target_port < target.port_count;
             ++target_port) {
          linked =
              linked || (target.ports[target_port].valid != 0u &&
                         target.ports[target_port].resource == row.resource &&
                         target.ports[target_port].access == 0u);
        }
        if (!linked) {
          return false;
        }
      }
      std::uint32_t expected_next = std::numeric_limits<std::uint32_t>::max();
      for (std::size_t next = index + 1u; next < proof.stage_count; ++next) {
        for (std::size_t next_port = 0u;
             next_port < proof.stages[next].port_count; ++next_port) {
          const auto &candidate = proof.stages[next].ports[next_port];
          if (candidate.valid != 0u && candidate.resource == row.resource &&
              candidate.access == 0u) {
            expected_next = static_cast<std::uint32_t>(next);
            break;
          }
        }
        if (expected_next != std::numeric_limits<std::uint32_t>::max()) {
          break;
        }
      }
      if (row.next_stage != expected_next) {
        return false;
      }
    }
    if (reads == 0u || writes != 1u || reads + writes != stage.port_count) {
      return false;
    }
    for (std::size_t port = 0u; port < reads; ++port) {
      if (!read_ports[port]) {
        return false;
      }
    }
    for (std::size_t port = 0u; port < writes; ++port) {
      if (!write_ports[port]) {
        return false;
      }
    }
  }
  return context.port_total == proof.port_count;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
