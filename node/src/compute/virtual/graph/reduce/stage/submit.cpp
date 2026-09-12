#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status StageController::submit(Ticket &ticket,
                               const ExecutionStage stage) noexcept {
  const bool prefix = stage == ExecutionStage::Prefix;
  const bool ready = prefix ? ticket.phase == TicketPhase::PrefixReady
                            : ticket.phase == TicketPhase::CollectiveReady;
  if (!ready || active_ != nullptr ||
      ticket.submitted != ExecutionStage::None) {
    return Status::fail(Reason::PipelineBusy);
  }
  const std::shared_ptr<PipelineState> &pipeline =
      prefix ? ticket.prefix : ticket.collective;
  const residency::EpochLease lease =
      prefix ? ticket.prefix_lease : ticket.collective_lease;
  const std::uint32_t stage_index =
      prefix ? 0u : static_cast<std::uint32_t>(terminal_stage_);
  WavefrontCoordinate selected{};
  if ((!prefix && !wavefront_.already_ready(ticket.batch, stage_index) &&
       !wavefront_.dependency_ready(ticket.batch, stage_index)) ||
      !wavefront_.select(selected) || selected.batch != ticket.batch ||
      selected.stage != stage_index || !wavefront_.dispatch(selected)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (pipeline == nullptr ||
      pipeline->residency_bank >= residency::Pool::BankCount ||
      !pool_.submit_execution(pipeline->residency_bank, pipeline, lease)) {
    (void)wavefront_.cancel_dispatch(selected);
    return Status::fail(Reason::PipelineBusy);
  }
  active_ = &ticket;
  ticket.submitted = stage;
  if (prefix) {
    ticket.prefix_executed = true;
    ticket.phase = TicketPhase::PrefixRunning;
  } else {
    ticket.collective_executed = true;
    ticket.phase = TicketPhase::CollectiveRunning;
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
