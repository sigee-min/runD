#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status CollectiveController::prepare(Ticket &ticket) noexcept {
  const auto note = [&](const Status failure, const Check check,
                        const residency::CloseInfo *const info = nullptr,
                        const std::uint64_t lease_token = 0u,
                        const std::uint64_t lease_generation = 0u) noexcept {
    failure_.note_cred(
        static_cast<std::uint32_t>(terminal_stage_), ticket.batch,
        Phase::Collective, check, failure, ticket.collective_epoch.ordinal,
        lease_token == 0u ? ticket.collective_receipt.token() : lease_token,
        lease_generation == 0u ? ticket.collective_receipt.generation()
                               : lease_generation,
        info);
  };
  if (ticket.phase != TicketPhase::IntermediateDirty ||
      !ticket.intermediate_dirty || ticket.collective_use_count == 0u ||
      ticket.collective_request_count == 0u ||
      ticket.collective_anchor_port >= ticket.collective_request_count ||
      ticket.collective_output_port >= ticket.collective_request_count) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Ticket);
    return failure;
  }
  const bool cpu = ticket.collective->device->backend == Backend::Cpu;
  CpuEpochPermit permit{};
  if (cpu) {
    const CpuReserveResult reserved =
        ticket.collective_receipt.reserve(authority_, permit);
    if (reserved != CpuReserveResult::Reserved) {
      const Status failure = reserved == CpuReserveResult::Busy
                                 ? Status::fail(Reason::PipelineBusy)
                                 : Status::fail(Reason::PipelineInvalid);
      note(failure, Check::Authority);
      return failure;
    }
    if (!permit.bound_to(authority_)) {
      const Status failure = Status::fail(Reason::PipelineInvalid);
      permit.cancel();
      note(failure, Check::Authority);
      return failure;
    }
  }
  const residency::AuthorityResult acquired = authority_.begin_graph_epoch(
      std::span<const residency::PageUse>{ticket.stage_uses.data(),
                                          ticket.collective_use_count},
      std::span<const residency::GraphPortRequest>{
          ticket.stage_requests.data(), ticket.collective_request_count},
      ticket.collective_anchor_port, ticket.collective_epoch.ordinal,
      cpu ? permit.key() : residency::CpuReservationKey{});
  if (!acquired) {
    permit.cancel();
    const Status failure = authority_status(acquired);
    note(failure, Check::Authority);
    return failure;
  }
  if (!cpu && (acquired.lease.token == 0u || acquired.lease.generation == 0u)) {
    permit.cancel();
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Authority, nullptr, acquired.lease.token,
         acquired.lease.generation);
    return failure;
  }
  if (cpu) {
    residency::CloseInfo bind_info{};
    const CpuBindResult bound =
        bind_cpu_epoch(authority_, permit, acquired.lease.token,
                       acquired.lease.generation, &bind_info);
    if (bound != CpuBindResult::Bound) {
      const Status failure = bound == CpuBindResult::Retained
                                 ? Status::fail(Reason::PipelineBusy)
                                 : Status::fail(Reason::PipelineInvalid);
      note(failure, Check::Authority, &bind_info, acquired.lease.token,
           acquired.lease.generation);
      return failure;
    }
  }
  ticket.collective_lease = acquired.lease;
  const auto rollback = [&](const bool invalidate) noexcept {
    residency::CloseInfo info{};
    const bool closed =
        cpu ? close_cpu_epoch(authority_, ticket.collective_receipt, false,
                              invalidate, &info)
            : authority_.complete(ticket.collective_lease.token, false, invalidate);
    if (!closed && cpu) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    return closed;
  };
  const Status relocated = relocate_graph_lease(*ticket.collective, graph_,
                                                pool_, acquired.lease, stats_);
  if (!relocated) {
    note(relocated, Check::Relocate);
    const bool terminal = rollback(true);
    if (terminal) {
      ticket.collective_lease = {};
    }
    ticket.poison = cpu ? (!terminal || ticket.poison) : true;
    return terminal ? relocated : Status::fail(Reason::PipelineInvalid);
  }
  if (!valid_stage_lease(ticket.collective_lease, ticket.count,
                         ticket.collective_request_count)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Collective);
    const bool terminal = rollback(true);
    if (terminal) {
      ticket.collective_lease = {};
    }
    return Status::fail(Reason::PipelineInvalid);
  }
  bool complete = true;
  for (std::size_t port_index = 0u;
       complete && port_index < ticket.collective_lease.ports.size(); ++port_index) {
    const residency::GraphLeasePort &port = ticket.collective_lease.ports[port_index];
    if (port.first_binding > ticket.collective_lease.bindings.size() ||
        port.binding_count != ticket.count ||
        port.binding_count >
            ticket.collective_lease.bindings.size() - port.first_binding) {
      complete = false;
      break;
    }
    const auto bindings = std::span<const residency::CacheBinding>{
        ticket.collective_lease.bindings.data() + port.first_binding,
        port.binding_count};
    if (port.access == residency::Access::Read) {
      complete = std::none_of(
          bindings.begin(), bindings.end(),
          [](const residency::CacheBinding binding) { return binding.fetch; });
    } else if (port.access == residency::Access::Write &&
               port_index == ticket.collective_output_port) {
      std::copy(bindings.begin(), bindings.end(),
                ticket.resident_outputs.begin());
    } else {
      complete = false;
    }
  }
  if (!complete) {
    note(Status::fail(Reason::PipelineInvalid), Check::Collective);
    const bool terminal = rollback(true);
    if (terminal) {
      ticket.collective_lease = {};
    }
    return terminal ? Status::fail(Reason::PipelineInvalid)
                    : Status::fail(Reason::PipelineBusy);
  }
  const residency::GraphLeasePort &anchor =
      ticket.collective_lease.ports[ticket.collective_anchor_port];
  if (anchor.access != residency::Access::Read ||
      anchor.first_binding > ticket.collective_lease.bindings.size() ||
      anchor.binding_count >
          ticket.collective_lease.bindings.size() - anchor.first_binding) {
    note(Status::fail(Reason::PipelineInvalid), Check::Collective);
    const bool terminal = rollback(true);
    if (terminal) {
      ticket.collective_lease = {};
    }
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto anchor_bindings = std::span<const residency::CacheBinding>{
      ticket.collective_lease.bindings.data() + anchor.first_binding,
      anchor.binding_count};
  VirtualTransferInterval control{};
  const Status controls =
      write_controls(*ticket.collective, run_, anchor_bindings, anchor.region,
                     stats_, &control);
  const bool control_recorded = record_interval(
      nullptr,
      Interval{.started = control.started_ns,
               .completed = control.completed_ns},
      Timeline::Direction::HostToDevice, stats_.pipeline.residency);
  if (!controls || !control_recorded ||
      !authority_.activate(ticket.collective_lease.token)) {
    note(!controls ? controls : Status::fail(Reason::PipelineInvalid),
         Check::Activate);
    const bool terminal = rollback(true);
    if (terminal) {
      ticket.collective_lease = {};
    }
    return !controls ? controls : Status::fail(Reason::PipelineInvalid);
  }
  ticket.phase = TicketPhase::CollectiveReady;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
