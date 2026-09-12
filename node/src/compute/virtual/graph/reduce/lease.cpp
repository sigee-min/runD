#include "lease.hpp"

namespace rund::compute::detail::graph_reduce {

bool valid_stage_lease(const residency::EpochLease &lease,
                       const std::size_t pages,
                       const std::size_t ports) noexcept {
  return lease.token != 0u && lease.generation != 0u && pages != 0u &&
         pages <= PipelineLeafCapacity && ports != 0u &&
         ports <= residency::TiledGraphPortCapacity &&
         lease.bindings.size() == pages * ports &&
         lease.ports.size() == ports &&
         lease.transitions.size() <=
             residency::TiledGraphPortCapacity * PipelineLeafCapacity * 3u &&
         lease.remaps.size() <= residency::GraphPageRemapCapacity;
}

residency::EpochLease prefix_input_lease(const Ticket &ticket) noexcept {
  const auto &lease = ticket.prefix_lease;
  if (lease.token == 0u || lease.ports.empty() ||
      lease.ports[0].access != residency::Access::Read ||
      lease.ports[0].first_binding != 0u ||
      lease.ports[0].binding_count != ticket.count ||
      ticket.count > lease.bindings.size()) {
    return {};
  }
  return residency::EpochLease{
      .bindings = lease.bindings.first(ticket.count),
      .transitions = lease.transitions,
      .token = lease.token,
      .generation = lease.generation,
  };
}

residency::EpochLease collective_output_lease(const Ticket &ticket) noexcept {
  const auto &lease = ticket.collective_lease;
  if (lease.token == 0u ||
      ticket.collective_output_port >= lease.ports.size()) {
    return {};
  }
  const residency::GraphLeasePort &output =
      lease.ports[ticket.collective_output_port];
  if (output.access != residency::Access::Write ||
      output.first_binding > lease.bindings.size() ||
      output.binding_count != ticket.count ||
      output.binding_count > lease.bindings.size() - output.first_binding) {
    return {};
  }
  return residency::EpochLease{
      .bindings =
          lease.bindings.subspan(output.first_binding, output.binding_count),
      .token = lease.token,
      .generation = lease.generation,
  };
}

} // namespace rund::compute::detail::graph_reduce
