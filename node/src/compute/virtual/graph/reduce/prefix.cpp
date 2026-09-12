#include "prefix.hpp"

#include "authority.hpp"
#include "failure.hpp"
#include "lease.hpp"
#include "projection.hpp"

namespace rund::compute::detail::graph_reduce {

Status finish_prefix(residency::Authority &authority,
                     const residency::TiledGraphPlan &graph,
                     Wavefront &wavefront, StageController &stages,
                     Ticket &ticket, FailLog &failure,
                     bool &child_poison) noexcept {
  const auto note = [&](const Status status, const Check check,
                        const residency::CloseInfo *const info = nullptr,
                        const std::uint64_t token = 0u,
                        const std::uint64_t generation = 0u) noexcept {
    failure.note_cred(0u, ticket.batch, Phase::Prefix, check, status,
                      ticket.prefix_epoch.ordinal,
                      token == 0u ? ticket.prefix_receipt.token() : token,
                      generation == 0u ? ticket.prefix_receipt.generation()
                                       : generation,
                      info);
  };
  const Status waited =
      stages.wait(ticket, ExecutionStage::Prefix, true, child_poison);
  if (!waited) {
    note(waited, Check::Wait);
    return waited;
  }
  const bool cpu = ticket.prefix != nullptr &&
                   ticket.prefix->device != nullptr &&
                   ticket.prefix->device->backend == Backend::Cpu;
  const auto rollback = [&](const bool invalidate,
                            residency::CloseInfo *const info =
                                nullptr) noexcept {
    return cpu ? close_cpu_epoch(authority, ticket.prefix_receipt, false,
                                 invalidate, info)
               : authority.complete(ticket.prefix_lease.token, false, invalidate);
  };
  const auto close = [&](residency::CloseInfo *const info = nullptr) noexcept {
    return cpu ? close_cpu_epoch(authority, ticket.prefix_receipt, true, false,
                                 info)
               : authority.complete(ticket.prefix_lease.token, true);
  };
  StageEffects effects{};
  if (!capture_stage_effects(graph, ticket.prefix_lease, effects)) {
    const std::uint64_t token = ticket.prefix_receipt.token();
    const std::uint64_t generation = ticket.prefix_receipt.generation();
    const Status failure_status = Status::fail(Reason::PipelineInvalid);
    note(failure_status, Check::Capture, nullptr, token, generation);
    residency::CloseInfo rollback_info{};
    const bool terminal = rollback(true, &rollback_info);
    if (rollback_info.check != residency::CloseInfo::Check::None) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &rollback_info,
           token, generation);
    }
    if (terminal) {
      ticket.prefix_lease = {};
    }
    child_poison = !terminal || child_poison;
    return failure_status;
  }
  const std::uint64_t token = ticket.prefix_receipt.token();
  const std::uint64_t generation = ticket.prefix_receipt.generation();
  residency::CloseInfo close_info{};
  if (!close(&close_info)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Close, &close_info,
         token, generation);
    residency::CloseInfo rollback_info{};
    const bool terminal = rollback(true, &rollback_info);
    if (rollback_info.check != residency::CloseInfo::Check::None) {
      note(Status::fail(Reason::PipelineBusy), Check::Recover, &rollback_info,
           token, generation);
    }
    if (terminal) {
      ticket.prefix_lease = {};
    }
    child_poison = cpu ? (!terminal || child_poison) : true;
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.prefix_lease = {};
  apply_stage_effects(ticket, effects);
  if (!wavefront.terminal(ticket.batch, 0u)) {
    note(Status::fail(Reason::PipelineInvalid), Check::Terminal, nullptr, token,
         generation);
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.intermediate_dirty = true;
  ticket.phase = TicketPhase::IntermediateDirty;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
