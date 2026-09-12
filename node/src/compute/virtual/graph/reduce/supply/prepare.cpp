#include "internal.hpp"

#include "../projection.hpp"
#include "../../../../device/residency/registry/graph_persist_owner.hpp"

namespace rund::compute::detail::graph_reduce {

Status SupplyController::prepare(Ticket &ticket, Timeline *const hidden_by,
                                 const bool prepared_before_ready) noexcept {
  const auto note = [&](const Status failure, const Check check,
                        const residency::CloseInfo *const info = nullptr,
                        const std::uint64_t lease_token = 0u,
                        const std::uint64_t lease_generation = 0u) noexcept {
    failure_.note_cred(
        0u, ticket.batch, Phase::Supply, check, failure,
        ticket.prefix_epoch.ordinal,
        lease_token == 0u ? ticket.prefix_receipt.token() : lease_token,
        lease_generation == 0u ? ticket.prefix_receipt.generation()
                               : lease_generation,
        info);
  };
  if (ticket.phase != TicketPhase::Projected) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Ticket);
    return failure;
  }
  const std::uint64_t started = pipeline_clock();
  if (ticket.prefix->device->backend != Backend::Cpu) {
    return prefetch_.consume(ticket, hidden_by);
  }

  CpuEpochPermit permit{};
  const CpuReserveResult reserved =
      ticket.prefix_receipt.reserve(authority_, permit);
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
  const auto uses = std::span<const residency::PageUse>{
      ticket.stage_uses.data(), ticket.prefix_use_count};
  const auto requests = std::span<const residency::GraphPortRequest>{
      ticket.stage_requests.data(), ticket.prefix_request_count};
  auto persist_owner = authority_.graph_persists();
  residency::AuthorityResult acquired{};
  if (retry_) {
    const std::size_t stage_count = graph_.stages().size();
    const residency::GraphPersistIdentity id =
        stage_count == 0u ? residency::GraphPersistIdentity{}
                          : persist_identity(run_, graph_, stage_count - 1u,
                                             ticket, prepared_);
    acquired = persist_owner.begin_cpu_graph_epoch_retry(
        uses, requests, ticket.prefix_anchor_port, ticket.prefix_epoch.ordinal,
        permit.key(), id);
  } else {
    acquired =
        authority_.begin_graph_epoch(uses, requests, ticket.prefix_anchor_port,
                                     ticket.prefix_epoch.ordinal, permit.key());
  }
  if (!acquired) {
    permit.cancel();
    const Status failure = authority_status(acquired);
    note(failure, Check::Authority);
    const bool recorded = record_interval(
        hidden_by, Interval{.started = started, .completed = pipeline_clock()},
        std::nullopt, stats_.pipeline.residency);
    return recorded ? failure : Status::fail(Reason::PipelineInvalid);
  }
  retry_ = false;
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
  ticket.prefix_lease = acquired.lease;
  const Status relocated = relocate_graph_lease(*ticket.prefix, graph_, pool_,
                                                acquired.lease, stats_);
  if (!relocated) {
    const Status failure = relocated;
    note(failure, Check::Relocate);
    residency::CloseInfo info{};
    const bool terminal =
        close_cpu_epoch(authority_, ticket.prefix_receipt, false, true, &info);
    if (!terminal) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    if (terminal) {
      ticket.prefix_lease = {};
    }
    ticket.poison = !terminal || ticket.poison;
    return terminal ? relocated : Status::fail(Reason::PipelineInvalid);
  }
  if (!valid_stage_lease(ticket.prefix_lease, ticket.count,
                         ticket.prefix_request_count)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Supply);
    (void)record_interval(
        hidden_by, Interval{.started = started, .completed = pipeline_clock()},
        std::nullopt, stats_.pipeline.residency);
    residency::CloseInfo info{};
    const bool terminal =
        close_cpu_epoch(authority_, ticket.prefix_receipt, false, true, &info);
    if (!terminal) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    if (terminal) {
      ticket.prefix_lease = {};
    }
    ticket.poison = !terminal || ticket.poison;
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto fail = [&](const Status status,
                        const Check check = Check::Supply) noexcept {
    note(status, check);
    residency::CloseInfo info{};
    const bool terminal =
        close_cpu_epoch(authority_, ticket.prefix_receipt, false, true, &info);
    if (!terminal) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    if (terminal) {
      ticket.prefix_lease = {};
    }
    ticket.poison = !terminal || ticket.poison;
    return terminal ? status : Status::fail(Reason::PipelineInvalid);
  };
  VirtualSupplyResult supplied{};
  const Status supplied_status = prefetch_.supply_cpu_stage(
      ticket, *ticket.prefix, 0u, ticket.prefix_lease, supplied);
  const Interval ready{.started = started, .completed = pipeline_clock()};
  const bool recorded = record_interval(hidden_by, ready, std::nullopt,
                                        stats_.pipeline.residency);
  if (!supplied_status || !recorded) {
    return fail(!supplied_status ? supplied_status
                                 : Status::fail(Reason::PipelineInvalid));
  }
  classify_backing(stats_,
                   std::span<const residency::PageUse>{
                       ticket.stage_uses.data(), ticket.count},
                   supplied.fetched_pages, prepared_before_ready, false);
  if (!record_input_evidence(stats_, run_, ticket.prefix->device->backend,
                             ticket.prefix_lease.ports, ticket.prefix_lease.bindings,
                             ticket.prefix_lease.transitions, supplied.fetched_pages,
                             supplied.backing_bytes)) {
    return fail(Status::fail(Reason::PipelineInvalid));
  }
  const Status controls =
      write_controls(*ticket.prefix, run_, prefix_input_lease(ticket).bindings,
                     ticket.input_region, stats_);
  if (!controls || !authority_.activate(ticket.prefix_lease.token)) {
    return fail(controls ? Status::fail(Reason::PipelineInvalid) : controls);
  }
  std::size_t ready_count = 0u;
  for (const residency::GraphLeasePort &port : ticket.prefix_lease.ports) {
    const residency::TiledGraphResource *const resource =
        graph_.resource(port.resource);
    if (port.access != residency::Access::Read || resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      continue;
    }
    if (!wavefront_.host_ready(ticket.batch, 0u, port.resource)) {
      return fail(Status::fail(Reason::PipelineInvalid));
    }
    ++ready_count;
  }
  if (ready_count == 0u) {
    return fail(Status::fail(Reason::PipelineInvalid));
  }
  ticket.phase = TicketPhase::SupplyReady;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
