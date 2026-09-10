#include "lease.hpp"

#include <algorithm>

namespace rund::compute::detail::graph_reduce {

bool retain_prefix_lease(Ticket &ticket,
                         const residency::AuthorityResult &result) noexcept {
  if (!result ||
      result.lease.bindings.size() !=
          ticket.count * ticket.prefix_request_count ||
      result.lease.bindings.size() > ticket.prefix_bindings.size() ||
      result.lease.ports.size() != ticket.prefix_request_count ||
      result.lease.ports.size() > ticket.prefix_ports.size() ||
      result.lease.transitions.size() > ticket.prefix_transitions.size() ||
      result.lease.remaps.size() > ticket.prefix_remaps.size()) {
    return false;
  }
  ticket.prefix_binding_count = result.lease.bindings.size();
  ticket.prefix_transition_count = result.lease.transitions.size();
  ticket.prefix_port_count = result.lease.ports.size();
  ticket.prefix_remap_count = result.lease.remaps.size();
  std::copy(result.lease.bindings.begin(), result.lease.bindings.end(),
            ticket.prefix_bindings.begin());
  std::copy(result.lease.transitions.begin(), result.lease.transitions.end(),
            ticket.prefix_transitions.begin());
  std::copy(result.lease.ports.begin(), result.lease.ports.end(),
            ticket.prefix_ports.begin());
  std::copy(result.lease.remaps.begin(), result.lease.remaps.end(),
            ticket.prefix_remaps.begin());
  return true;
}

bool retain_collective_lease(
    Ticket &ticket, const residency::AuthorityResult &result) noexcept {
  if (!result ||
      result.lease.bindings.size() !=
          ticket.count * ticket.collective_request_count ||
      result.lease.bindings.size() > ticket.collective_bindings.size() ||
      result.lease.ports.size() != ticket.collective_request_count ||
      result.lease.ports.size() > ticket.collective_ports.size() ||
      result.lease.transitions.size() > ticket.collective_transitions.size() ||
      result.lease.remaps.size() > ticket.collective_remaps.size()) {
    return false;
  }
  ticket.collective_binding_count = result.lease.bindings.size();
  ticket.collective_transition_count = result.lease.transitions.size();
  ticket.collective_port_count = result.lease.ports.size();
  ticket.collective_remap_count = result.lease.remaps.size();
  std::copy(result.lease.bindings.begin(), result.lease.bindings.end(),
            ticket.collective_bindings.begin());
  std::copy(result.lease.transitions.begin(), result.lease.transitions.end(),
            ticket.collective_transitions.begin());
  std::copy(result.lease.ports.begin(), result.lease.ports.end(),
            ticket.collective_ports.begin());
  std::copy(result.lease.remaps.begin(), result.lease.remaps.end(),
            ticket.collective_remaps.begin());
  return true;
}

residency::EpochLease prefix_lease(Ticket &ticket) noexcept {
  return residency::EpochLease{
      .bindings =
          std::span<const residency::CacheBinding>{
              ticket.prefix_bindings.data(), ticket.prefix_binding_count},
      .transitions =
          std::span<const residency::CacheTransition>{
              ticket.prefix_transitions.data(), ticket.prefix_transition_count},
      .ports =
          std::span<const residency::GraphLeasePort>{ticket.prefix_ports.data(),
                                                     ticket.prefix_port_count},
      .remaps =
          std::span<const residency::GraphPageRemap>{
              ticket.prefix_remaps.data(), ticket.prefix_remap_count},
      .token = ticket.prefix_token,
  };
}

residency::EpochLease prefix_input_lease(Ticket &ticket) noexcept {
  if (ticket.prefix_port_count == 0u ||
      ticket.prefix_ports[0].access != residency::Access::Read ||
      ticket.prefix_ports[0].first_binding != 0u ||
      ticket.prefix_ports[0].binding_count != ticket.count) {
    return {};
  }
  return residency::EpochLease{
      .bindings =
          std::span<const residency::CacheBinding>{
              ticket.prefix_bindings.data(), ticket.count},
      .transitions =
          std::span<const residency::CacheTransition>{
              ticket.prefix_transitions.data(), ticket.prefix_transition_count},
      .token = ticket.prefix_token,
  };
}

residency::EpochLease collective_lease(Ticket &ticket) noexcept {
  return residency::EpochLease{
      .bindings =
          std::span<const residency::CacheBinding>{
              ticket.collective_bindings.data(),
              ticket.collective_binding_count},
      .transitions =
          std::span<const residency::CacheTransition>{
              ticket.collective_transitions.data(),
              ticket.collective_transition_count},
      .ports =
          std::span<const residency::GraphLeasePort>{
              ticket.collective_ports.data(), ticket.collective_port_count},
      .remaps =
          std::span<const residency::GraphPageRemap>{
              ticket.collective_remaps.data(), ticket.collective_remap_count},
      .token = ticket.collective_token,
  };
}

residency::EpochLease collective_output_lease(Ticket &ticket) noexcept {
  if (ticket.collective_output_port >= ticket.collective_port_count) {
    return {};
  }
  const residency::GraphLeasePort &output =
      ticket.collective_ports[ticket.collective_output_port];
  if (output.access != residency::Access::Write ||
      output.first_binding > ticket.collective_binding_count ||
      output.binding_count != ticket.count ||
      output.binding_count >
          ticket.collective_binding_count - output.first_binding) {
    return {};
  }
  return residency::EpochLease{
      .bindings =
          std::span<const residency::CacheBinding>{
              ticket.collective_bindings.data() + output.first_binding,
              output.binding_count},
      .token = ticket.collective_token,
  };
}

} // namespace rund::compute::detail::graph_reduce
