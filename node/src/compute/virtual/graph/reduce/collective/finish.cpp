#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status CollectiveController::finish(Ticket &ticket,
                                    bool &child_poison) noexcept {
  const auto note = [&](const Status failure, const Check check,
                        const residency::CloseInfo *const info =
                            nullptr) noexcept {
    failure_.note_cred(static_cast<std::uint32_t>(terminal_stage_),
                       ticket.batch, Phase::Collective, check, failure,
                       ticket.collective_epoch.ordinal,
                       ticket.collective_receipt.token(),
                       ticket.collective_receipt.generation(), info);
  };
  const Status waited =
      stages_.wait(ticket, ExecutionStage::Collective, false, child_poison);
  if (!waited) {
    note(waited, Check::Wait);
    return waited;
  }
  const bool cpu = ticket.collective != nullptr &&
                   ticket.collective->device != nullptr &&
                   ticket.collective->device->backend == Backend::Cpu;
  const auto rollback = [&](const bool invalidate,
                            residency::CloseInfo *const info =
                                nullptr) noexcept {
    return cpu ? close_cpu_epoch(authority_, ticket.collective_receipt, false,
                                 invalidate, info)
               : authority_.complete(ticket.collective_lease.token, false,
                                     invalidate);
  };
  const auto close = [&](residency::CloseInfo *const info = nullptr) noexcept {
    return cpu ? close_cpu_epoch(authority_, ticket.collective_receipt, true,
                                 false, info)
               : authority_.complete(ticket.collective_lease.token, true);
  };
  const Status retained = retain_residency_output(
      *ticket.collective, run_, collective_output_lease(ticket), stats_);
  if (!retained) {
    note(retained, Check::Capture);
    const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
    if (!folded) {
      note(folded, Check::Terminal);
    }
    child_poison = true;
    return folded ? retained : folded;
  }
  StageEffects effects{};
  if (!capture_stage_effects(graph_, ticket.collective_lease, effects)) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Capture);
    const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
    residency::CloseInfo info{};
    const bool terminal = rollback(true, &info);
    if (terminal) {
      ticket.collective_lease = {};
    } else {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    child_poison = !terminal || child_poison;
    return folded ? Status::fail(Reason::PipelineInvalid) : folded;
  }
  std::size_t output_index = 0u;
  if (!graph_resource_index(
          graph_, graph_.stages()[terminal_stage_].ports.back().resource,
          output_index)) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Collective);
    const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
    residency::CloseInfo info{};
    const bool terminal = rollback(true, &info);
    if (terminal) {
      ticket.collective_lease = {};
    } else {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &info);
    }
    child_poison = !terminal || child_poison;
    return folded ? Status::fail(Reason::PipelineInvalid) : folded;
  }
  effects.written[output_index] = false;
  residency::CloseInfo close_info{};
  if (!close(&close_info)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Close, &close_info);
    const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
    residency::CloseInfo rollback_info{};
    const bool terminal = rollback(true, &rollback_info);
    if (terminal) {
      ticket.collective_lease = {};
    } else {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &rollback_info);
    }
    child_poison = cpu ? (!terminal || child_poison) : true;
    return folded ? Status::fail(Reason::PipelineInvalid) : folded;
  }
  ticket.collective_lease = {};
  apply_stage_effects(ticket, effects);
  ticket.intermediate_dirty = false;
  ticket.output_dirty = true;
  if (!wavefront_.terminal(ticket.batch,
                           static_cast<std::uint32_t>(terminal_stage_))) {
    note(Status::fail(Reason::PipelineInvalid), Check::Terminal);
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.phase = TicketPhase::DeviceOutputDirty;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
