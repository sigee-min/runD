#include "internal.hpp"

#include "../../../backing.hpp"
#include "../../../run/backing.hpp"
#include "../transfer.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::supply_stage(Ticket &ticket,
                                        const StageScratch &scratch,
                                        const WavefrontCoordinate &coordinate,
                                        const std::uint32_t resource,
                                        bool &cleanup_failed) noexcept {
  const auto note = [&](const Status failure, const Check check,
                        const residency::CloseInfo *const info = nullptr,
                        const std::uint64_t lease_token = 0u,
                        const std::uint64_t lease_generation = 0u) noexcept {
    state_.failure_log.note_cred(
        coordinate.stage, ticket.batch, Phase::Supply, check, failure,
        coordinate.ordinal,
        lease_token == 0u ? ticket.supply_receipt.token() : lease_token,
        lease_generation == 0u ? ticket.supply_receipt.generation()
                               : lease_generation,
        info);
  };
  if (state_.pipeline->device->backend == Backend::Cpu) {
    if (ticket.input_promote || coordinate.stage >= graph_.stages().size() ||
        scratch.port_count == 0u || ticket.count == 0u) {
      const Status failure = Status::fail(Reason::PipelineInvalid);
      note(failure, Check::Ticket);
      return failure;
    }
    std::size_t selected_port = scratch.port_count;
    for (std::size_t index = 0u; index < scratch.port_count; ++index) {
      const residency::TiledGraphPort port =
          graph_.stages()[coordinate.stage].ports[index];
      if (port.resource == resource && port.access == residency::Access::Read) {
        if (selected_port != scratch.port_count) {
          const Status failure = Status::fail(Reason::PipelineInvalid);
          note(failure, Check::Supply);
          return failure;
        }
        selected_port = index;
      }
    }
    const std::shared_ptr<PipelineState> pipeline =
        graph_stage_pipeline(state_, coordinate.stage, ticket.bank);
    if (selected_port == scratch.port_count || pipeline == nullptr) {
      const Status failure = Status::fail(Reason::PipelineInvalid);
      note(failure, Check::Supply);
      return failure;
    }
    const residency::GraphPortRequest &request =
        scratch.requests[selected_port];
    std::array<residency::CacheUse, PipelineLeafCapacity> uses{};
    for (std::size_t page = 0u; page < ticket.count; ++page) {
      const residency::PageUse &use =
          scratch.uses[selected_port * ticket.count + page];
      residency::CacheKey key{};
      if (!residency::project_graph_cache_key(request.materialization, use.key,
                                              key)) {
        const Status failure = Status::fail(Reason::PipelineInvalid);
        note(failure, Check::Supply);
        return failure;
      }
      uses[page] = residency::CacheUse{.key = key,
                                       .access = residency::Access::Read,
                                       .next_use = use.next_use,
                                       .epoch = use.prefetch_epoch,
                                       .retain_until = use.pin.last_epoch};
    }
    CpuEpochPermit permit{};
    const CpuReserveResult reserved =
        ticket.supply_receipt.reserve(authority_, permit);
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
    const residency::AuthorityResult acquired = authority_.begin(
        std::span<const residency::CacheUse>{uses.data(), ticket.count},
        request.region.tier, request.region.role, request.region.first,
        request.region.count, permit.key());
    if (!acquired) {
      permit.cancel();
      const Status failure = authority_status(acquired);
      note(failure, Check::Authority);
      return failure;
    }
    const std::array<residency::GraphLeasePort, 1u> ports{
        residency::GraphLeasePort{
            .program_port = request.program_port,
            .access = residency::Access::Read,
            .resource = resource,
            .region = request.region,
            .cache_regions = request.cache_regions,
            .cache_region_count = request.cache_region_count,
            .first_binding = 0u,
            .binding_count = ticket.count,
        }};
    const residency::EpochLease lease{
        .bindings = acquired.lease.bindings,
        .transitions = acquired.lease.transitions,
        .ports = ports,
        .relocations = acquired.lease.relocations,
        .token = acquired.lease.token,
        .generation = acquired.lease.generation,
    };
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
    const auto fail = [&](const Status failure, const bool invalidate,
                          const Check check) noexcept {
      note(failure, check);
      residency::CloseInfo info{};
      const bool terminal = close_cpu_epoch(authority_, ticket.supply_receipt,
                                            false, invalidate, &info);
      if (!terminal) {
        note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
      }
      cleanup_failed = !terminal || cleanup_failed;
      return terminal ? failure : Status::fail(Reason::PipelineBusy);
    };
    const Status relocated =
        relocate_graph_lease(*pipeline, graph_, pool_, lease, stats_);
    if (!relocated) {
      return fail(relocated, true, Check::Relocate);
    }
    VirtualSupplyResult supplied{};
    const Status supplied_status =
        supply_cpu_stage(ticket, *pipeline, coordinate.stage, lease, supplied);
    if (!supplied_status || !authority_.activate(lease.token)) {
      return fail(supplied_status ? Status::fail(Reason::PipelineInvalid)
                                  : supplied_status,
                  true, supplied_status ? Check::Activate : Check::Supply);
    }
    residency::CloseInfo close_info{};
    if (!close_cpu_epoch(authority_, ticket.supply_receipt, true, false,
                         &close_info)) {
      note(Status::fail(Reason::PipelineInvalid), Check::Close, &close_info);
      residency::CloseInfo rollback_info{};
      const bool terminal = close_cpu_epoch(authority_, ticket.supply_receipt,
                                            false, true, &rollback_info);
      if (!terminal) {
        note(Status::fail(Reason::PipelineBusy), Check::Recover,
             &rollback_info);
      }
      cleanup_failed = !terminal || cleanup_failed;
      return terminal ? Status::fail(Reason::PipelineInvalid)
                      : Status::fail(Reason::PipelineBusy);
    }
    using ::rund::detail::counter::Accumulate;
    Accumulate(stats_.pipeline.residency.page_in_count, supplied.fetched_pages);
    Accumulate(stats_.pipeline.residency.cache_hit_count,
               ticket.count - supplied.fetched_pages);
    if (!wavefront_.host_ready(ticket.batch, coordinate.stage, resource)) {
      note(Status::fail(Reason::PipelineInvalid), Check::Activate);
      ticket.poison = true;
      cleanup_failed = true;
      return Status::fail(Reason::PipelineInvalid);
    }
    const bool ready = wavefront_.device_ready(ticket.batch, coordinate.stage);
    if (!ready) {
      std::size_t external = 0u;
      for (const residency::TiledGraphPort port :
           graph_.stages()[coordinate.stage].ports) {
        const residency::TiledGraphResource *const declared =
            graph_.resource(port.resource);
        external += static_cast<std::size_t>(
            port.access == residency::Access::Read && declared != nullptr &&
            declared->kind == residency::GraphResourceKind::ExternalInput &&
            declared->persistence == residency::ResourcePersistence::Backing);
      }
      if (external <= 1u &&
          !wavefront_.already_ready(ticket.batch, coordinate.stage)) {
        ticket.poison = true;
        cleanup_failed = true;
        note(Status::fail(Reason::PipelineInvalid), Check::Activate);
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    return Status::success();
  }
  if (ticket.input_promote) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PrefetchLane &selected = lane(ticket.batch);
  const std::size_t selected_index = lane_index(ticket.batch);
  if (selected.pending &&
      (selected.batch == ticket.batch || !selected.speculative ||
       selected.stage != 0u || !retire(selected_index, selected))) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!pool_.prefetch[selected_index].quiescent()) {
    return Status::fail(Reason::PipelineBusy);
  }

  const residency::PoolPhysicalOwner *input_owner = nullptr;
  Status status = project_stage(selected, ticket, scratch, coordinate, resource,
                                input_owner);
  selected.physical_lane = static_cast<std::uint32_t>(selected_index);
  if (status) {
    status = select_missing(selected, *input_owner, false);
  }
  if (!status) {
    selected = {};
    return status;
  }
  if (selected.device_only) {
    selected.pending = true;
    const Status consumed = consume(ticket, coordinate.stage, nullptr);
    if (!consumed) {
      return consumed;
    }
    WavefrontCoordinate promotable{};
    if (ticket.host_ready_count == 0u &&
        wavefront_.promote(ticket.batch, promotable) &&
        promotable == coordinate &&
        !wavefront_.device_ready(ticket.batch, coordinate.stage)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    return Status::success();
  }

  const residency::TiledGraphPort input{.resource = resource,
                                        .access = residency::Access::Read};
  status = issue(lane_index(ticket.batch), selected, input, cleanup_failed);
  if (status) {
    status = consume(ticket, coordinate.stage, nullptr);
  }
  return status;
}

} // namespace rund::compute::detail::graph_reduce
