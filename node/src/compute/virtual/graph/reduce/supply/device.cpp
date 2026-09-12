#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status SupplyController::make_device_ready(Ticket &ticket,
                                           Timeline *const hidden_by) noexcept {
  if (ticket.prefix->device->backend == Backend::Cpu) {
    if (ticket.phase != TicketPhase::SupplyReady ||
        !wavefront_.device_ready(ticket.batch, 0u)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    ticket.phase = TicketPhase::PrefixReady;
    return Status::success();
  }
  if (ticket.phase != TicketPhase::SupplyReady ||
      ticket.host_supply.page_count > ticket.host_supply.pages.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::AuthorityResult acquired = authority_.begin_graph_epoch(
      std::span<const residency::PageUse>{ticket.stage_uses.data(),
                                          ticket.prefix_use_count},
      std::span<const residency::GraphPortRequest>{
          ticket.stage_requests.data(), ticket.prefix_request_count},
      ticket.prefix_anchor_port, ticket.prefix_epoch.ordinal);
  if (!acquired) {
    return authority_status(acquired);
  }
  ticket.prefix_lease = acquired.lease;
  const Status relocated = relocate_graph_lease(*ticket.prefix, graph_, pool_,
                                                acquired.lease, stats_);
  if (!relocated) {
    const bool terminal = authority_.complete(ticket.prefix_lease.token, false, true);
    ticket.prefix_lease = {};
    ticket.poison = true;
    return terminal ? relocated : Status::fail(Reason::PipelineInvalid);
  }
  if (!valid_stage_lease(ticket.prefix_lease, ticket.count,
                         ticket.prefix_request_count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::EpochLease inputs = prefix_input_lease(ticket);
  std::uint64_t execution_fetches = 0u;
  for (std::size_t port_index = 0u; port_index < ticket.prefix_lease.ports.size();
       ++port_index) {
    const residency::GraphLeasePort port = ticket.prefix_lease.ports[port_index];
    if (port.access != residency::Access::Read ||
        port.first_binding > ticket.prefix_lease.bindings.size() ||
        port.binding_count > ticket.prefix_lease.bindings.size() - port.first_binding) {
      if (port.access == residency::Access::Read) {
        return Status::fail(Reason::PipelineInvalid);
      }
      continue;
    }
    for (std::size_t page = 0u; page < port.binding_count; ++page) {
      execution_fetches += static_cast<std::uint64_t>(
          ticket.prefix_lease.bindings[port.first_binding + page].fetch);
    }
  }
  VirtualTransferInterval control{};
  const Status controls = write_controls(*ticket.prefix, run_, inputs.bindings,
                                         ticket.input_region, stats_, &control);
  const bool control_recorded =
      !controls || record_interval(hidden_by,
                                   Interval{.started = control.started_ns,
                                            .completed = control.completed_ns},
                                   Timeline::Direction::HostToDevice,
                                   stats_.pipeline.residency);
  if (!controls || !control_recorded) {
    return !controls ? controls : Status::fail(Reason::PipelineInvalid);
  }

  if (ticket.host_ready_count != 0u) {
    const Status issued = issue_input_promotion(
        authority_, run_.active.graph, ticket, 0u, ticket.prefix_lease.token);
    if (!issued) {
      return issued;
    }
    const InputPromotionResult promoted = transfer_input_promotion(
        authority_, *ticket.prefix, run_, ticket.prefix_lease, stats_, ticket);
    const bool upload_recorded =
        !promoted.status || execution_fetches == 0u ||
        record_interval(hidden_by, promoted.interval,
                        Timeline::Direction::HostToDevice,
                        stats_.pipeline.residency);
    if (!promoted.released) {
      ticket.poison = true;
    }
    if (!promoted.status || !upload_recorded || !promoted.released) {
      return !promoted.status ? promoted.status
                              : Status::fail(Reason::PipelineInvalid);
    }
  } else {
    VirtualTransferInterval upload{};
    const Status uploaded = supply_residency_cache(*ticket.prefix, run_, inputs,
                                                   stats_, nullptr, &upload);
    const bool upload_recorded =
        !uploaded || execution_fetches == 0u ||
        record_interval(hidden_by,
                        Interval{.started = upload.started_ns,
                                 .completed = upload.completed_ns},
                        Timeline::Direction::HostToDevice,
                        stats_.pipeline.residency);
    if (!uploaded || !upload_recorded || execution_fetches != 0u ||
        !authority_.activate(ticket.prefix_lease.token)) {
      return !uploaded ? uploaded : Status::fail(Reason::PipelineInvalid);
    }
  }
  if (!record_input_evidence(stats_, run_, ticket.prefix->device->backend,
                             ticket.prefix_lease.ports, ticket.prefix_lease.bindings,
                             ticket.prefix_lease.transitions,
                             ticket.host_supply.fetched_pages,
                             ticket.host_supply.backing_bytes)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const bool device_ready = ticket.host_supply.page_count == 0u
                                ? wavefront_.device_resident(ticket.batch, 0u)
                                : wavefront_.device_ready(ticket.batch, 0u);
  if (!device_ready) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.phase = TicketPhase::PrefixReady;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
