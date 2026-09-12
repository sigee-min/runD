#include "internal.hpp"

#include "../evidence.hpp"

#include <rund/compute/pipeline/runtime.hpp>

namespace rund::compute::detail::graph_reduce {

Status
MiddleController::run_stage(Ticket &ticket, const StageScratch &scratch,
                            const WavefrontCoordinate &selected,
                            const std::shared_ptr<PipelineState> &pipeline,
                            bool &child_poison) noexcept {
  using ::rund::detail::counter::Accumulate;
  const auto note = [&](const Status failure, const Check check,
                        const residency::CloseInfo *const info = nullptr,
                        const std::uint64_t lease_token = 0u,
                        const std::uint64_t lease_generation = 0u) noexcept {
    state_.failure_log.note_cred(
        selected.stage, selected.batch, Phase::Middle, check, failure,
        selected.ordinal,
        lease_token == 0u ? ticket.middle_receipt.token() : lease_token,
        lease_generation == 0u ? ticket.middle_receipt.generation()
                               : lease_generation,
        info);
  };
  const bool cpu = pipeline != nullptr && pipeline->device != nullptr &&
                   pipeline->device->backend == Backend::Cpu;
  CpuEpochPermit permit{};
  if (cpu) {
    const CpuReserveResult reserved =
        ticket.middle_receipt.reserve(authority_, permit);
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
      std::span<const residency::PageUse>{scratch.uses.data(),
                                          scratch.use_count},
      std::span<const residency::GraphPortRequest>{scratch.requests.data(),
                                                   scratch.port_count},
      scratch.anchor_port, scratch.epoch.ordinal,
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
  const std::uint64_t token = acquired.lease.token;
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
  const auto rollback = [&](const bool invalidate) noexcept {
    residency::CloseInfo info{};
    const bool terminal =
        cpu ? close_cpu_epoch(authority_, ticket.middle_receipt, false,
                              invalidate, &info)
            : authority_.complete(token, false, invalidate);
    if (!terminal) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover,
           cpu ? &info : nullptr);
    }
    return terminal;
  };
  const auto close = [&]() noexcept {
    residency::CloseInfo info{};
    const bool terminal =
        cpu ? close_cpu_epoch(authority_, ticket.middle_receipt, true, false,
                              &info)
            : authority_.complete(token, true);
    if (!terminal) {
      note(Status::fail(Reason::PipelineInvalid), Check::Close,
           cpu ? &info : nullptr);
    }
    return terminal;
  };
  const Status relocated =
      relocate_graph_lease(*pipeline, graph_, pool_, acquired.lease, stats_);
  if (!relocated) {
    note(relocated, Check::Relocate);
    const bool terminal = rollback(true);
    child_poison = cpu ? (!terminal || child_poison) : true;
    return terminal ? relocated : Status::fail(Reason::PipelineInvalid);
  }
  const bool promoting = ticket.host_ready_count != 0u;
  if (promoting && (ticket.forecast_stage != selected.stage ||
                    ticket.host_supply.page_count == 0u)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Promote);
    const bool terminal = rollback(true);
    child_poison = !terminal || child_poison;
    return Status::fail(Reason::PipelineInvalid);
  }
  bool supplied = true;
  for (const residency::GraphLeasePort &port : acquired.lease.ports) {
    if (port.access != residency::Access::Read) {
      continue;
    }
    for (std::size_t page = 0u; page < port.binding_count; ++page) {
      if (port.first_binding + page >= acquired.lease.bindings.size() ||
          acquired.lease.bindings[port.first_binding + page].fetch) {
        supplied = false;
        break;
      }
    }
  }
  const residency::GraphLeasePort &anchor =
      acquired.lease.ports[scratch.anchor_port];
  const auto anchor_bindings =
      anchor.first_binding <= acquired.lease.bindings.size() &&
              anchor.binding_count <=
                  acquired.lease.bindings.size() - anchor.first_binding
          ? acquired.lease.bindings.subspan(anchor.first_binding,
                                            anchor.binding_count)
          : std::span<const residency::CacheBinding>{};
  VirtualTransferInterval control{};
  const Status controls = (supplied || promoting)
                              ? write_controls(*pipeline, run_, anchor_bindings,
                                               anchor.region, stats_, &control)
                              : Status::fail(Reason::PipelineInvalid);
  const bool control_recorded =
      !controls || record_interval(nullptr,
                                   Interval{.started = control.started_ns,
                                            .completed = control.completed_ns},
                                   Timeline::Direction::HostToDevice,
                                   stats_.pipeline.residency);
  if (!controls || !control_recorded) {
    note(!controls ? controls : Status::fail(Reason::PipelineInvalid),
         Check::Activate);
    const bool terminal = rollback(false);
    child_poison = !terminal || child_poison;
    return !controls ? controls : Status::fail(Reason::PipelineInvalid);
  }
  if (promoting) {
    const Status issued = issue_input_promotion(authority_, run_.active.graph,
                                                ticket, selected.stage, token);
    if (!issued) {
      note(issued, Check::Promote);
      const bool terminal = rollback(true);
      child_poison = !terminal || child_poison;
      return issued;
    }
    const InputPromotionResult promoted = transfer_input_promotion(
        authority_, *pipeline, run_, acquired.lease, stats_, ticket);
    const bool upload_recorded =
        !promoted.status || record_interval(nullptr, promoted.interval,
                                            Timeline::Direction::HostToDevice,
                                            stats_.pipeline.residency);
    if (!promoted.status || !upload_recorded || !promoted.released) {
      note(!promoted.status ? promoted.status
                            : Status::fail(Reason::PipelineInvalid),
           Check::Promote);
      const bool terminal = rollback(true);
      child_poison = !promoted.released || !terminal || child_poison;
      if (!terminal) {
        return Status::fail(Reason::PipelineBusy);
      }
      return !promoted.status ? promoted.status
                              : Status::fail(Reason::PipelineInvalid);
    }
    // Preserve the Middle promotion counters: read hits/fetches only.
    // This observation has never folded lease transitions into evictions.
    if (!record_input_evidence(stats_, run_, pipeline->device->backend,
                               acquired.lease.ports, acquired.lease.bindings,
                               {}, 0u, 0u)) {
      note(Status::fail(Reason::PipelineInvalid), Check::Capture);
      const bool terminal = rollback(true);
      child_poison = true;
      (void)terminal;
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!wavefront_.device_ready(ticket.batch, selected.stage)) {
      note(Status::fail(Reason::PipelineInvalid), Check::Activate);
      const bool terminal = rollback(true);
      child_poison = !terminal || child_poison;
      return Status::fail(Reason::PipelineInvalid);
    }
  } else if (!supplied || !authority_.activate(token)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Activate);
    const bool terminal = rollback(false);
    child_poison = !terminal || child_poison;
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!wavefront_.dispatch(selected)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Execute);
    const bool terminal = rollback(true);
    child_poison = !terminal || child_poison;
    return Status::fail(Reason::PipelineInvalid);
  }
  if (pipeline == nullptr ||
      pipeline->residency_bank >= residency::Pool::BankCount ||
      !pool_.submit_execution(pipeline->residency_bank, pipeline,
                              acquired.lease)) {
    note(Status::fail(Reason::PipelineBusy), Check::Submit);
    const bool terminal = rollback(false);
    child_poison = !terminal || child_poison;
    return Status::fail(Reason::PipelineBusy);
  }
  const residency::ExecutionReceipt execution =
      pool_.wait_execution(pipeline->residency_bank);
  const Status folded = fold_stage(stats_, pipeline, identity_);
  if (!execution.status) {
    note(execution.status, Check::Wait);
    const bool terminal = rollback(true);
    child_poison =
        !terminal || !folded || poisoned_pipeline(pipeline) || child_poison;
    return folded ? execution.status : folded;
  }
  StageEffects effects{};
  const bool captured = capture_stage_effects(graph_, acquired.lease, effects);
  if (!folded || !captured) {
    note(!folded ? folded : Status::fail(Reason::PipelineInvalid),
         Check::Capture);
    const bool terminal = rollback(true);
    child_poison = !terminal || child_poison;
    return !folded ? folded : Status::fail(Reason::PipelineInvalid);
  }
  if (!close()) {
    const bool terminal = rollback(true);
    child_poison = cpu ? (!terminal || child_poison) : true;
    return Status::fail(Reason::PipelineInvalid);
  }
  apply_stage_effects(ticket, effects);
  if (!wavefront_.terminal(selected)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Terminal);
    child_poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  Accumulate(stats_.pipeline.residency.epoch_count, 1u);
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
