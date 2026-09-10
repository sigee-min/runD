#include "../persist.hpp"

#include "../../timeline.hpp"

namespace rund::compute::detail::graph_reduce {

Status PersistController::prepare_host_output(Ticket &ticket,
                                              Timeline *const hidden_by,
                                              bool &child_poison) noexcept {
  if (ticket.collective->device->backend == Backend::Cpu) {
    const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
    if (!folded) {
      child_poison = true;
      return folded;
    }
    ticket.phase = TicketPhase::HostOutputDirty;
    return Status::success();
  }

  const OutputDrainResult drained =
      transfer_output_drain(authority_, run_, ticket);
  const bool recorded = !drained.transfer_complete ||
                        record_interval(hidden_by, drained.interval,
                                        Timeline::Direction::DeviceToHost,
                                        stats_.pipeline.residency);
  const Status folded = stages_.fold(ticket, ExecutionStage::Collective);
  if (!drained.transfer_complete || !folded || !recorded || !drained.released) {
    ticket.output_dirty = !drained.released || drained.transfer_complete;
    if (drained.released && drained.transfer_complete) {
      ticket.phase = TicketPhase::HostOutputDirty;
    }
    child_poison = !folded || !recorded || !drained.released || child_poison;
    return !drained.status
               ? drained.status
               : (!folded ? folded : Status::fail(Reason::CompletionInvalid));
  }

  ticket.phase = TicketPhase::HostOutputDirty;
  // GraphDrain callback-return release retires the Device output bank. The
  // disjoint Host output bank remains owned by GraphPersist until its own
  // callback-return release, so publishing the Device reuse fact here does
  // not permit same-bank Host storage reuse.
  if (!wavefront_.release(ticket.batch)) {
    child_poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
