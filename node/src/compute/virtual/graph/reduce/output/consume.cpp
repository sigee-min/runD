#include "../output.hpp"

#include "../../../run/reduce.hpp"
#include "../authority.hpp"
#include "../timeline.hpp"

namespace rund::compute::detail::graph_reduce {

Status finish_output(residency::Authority &authority,
                     const VirtualRunProjection &run,
                     VirtualReduction &reduction, Stats &stats,
                     Wavefront &wavefront, StageController &stages,
                     Ticket &ticket, Timeline *const hidden_by,
                     bool &child_poison) noexcept {
  if (!ticket.output_dirty || ticket.intermediate_dirty) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (ticket.collective->device->backend != Backend::Cpu) {
    if (ticket.phase != TicketPhase::OutputDraining || !ticket.output_drain) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const OutputDrainResult drained =
        transfer_output_drain(authority, run, ticket);
    const bool recorded = !drained.transfer_complete ||
                          record_interval(hidden_by, drained.interval,
                                          Timeline::Direction::DeviceToHost,
                                          stats.pipeline.residency);
    const Status folded = stages.fold(ticket, ExecutionStage::Collective);
    if (!drained.transfer_complete || !folded || !recorded) {
      // A successful physical transfer followed by a logical/timeline failure
      // leaves the Host row Dirty. Abort cleanup must discard that exact Host
      // frame. A failed released transfer already discarded both rows.
      ticket.output_dirty = !drained.released || drained.transfer_complete;
      if (drained.released && drained.transfer_complete) {
        ticket.phase = TicketPhase::HostOutputDirty;
      }
      child_poison = !folded || !recorded || !drained.released || child_poison;
      return !drained.status
                 ? drained.status
                 : (!folded ? folded : Status::fail(Reason::CompletionInvalid));
    }
    if (!drained.released) {
      child_poison = true;
      return Status::fail(Reason::CompletionInvalid);
    }
    ticket.phase = TicketPhase::HostOutputDirty;
  } else {
    if (ticket.phase != TicketPhase::DeviceOutputDirty) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status folded = stages.fold(ticket, ExecutionStage::Collective);
    if (!folded) {
      child_poison = true;
      return folded;
    }
  }

  const residency::EpochLease reduction_lease{
      .bindings =
          std::span<const residency::CacheBinding>{
              ticket.resident_outputs.data(), ticket.count},
  };
  const Status consumed =
      consume_virtual_reduction(run, reduction_lease, reduction);
  const bool output_clean =
      discard_keys(authority,
                   std::span<const residency::CacheKey>{
                       ticket.output_keys.data(), ticket.count},
                   ticket.resident_output_region);
  if (output_clean) {
    ticket.output_dirty = false;
  }
  if (!consumed || !output_clean) {
    child_poison = !output_clean || child_poison;
    return consumed ? Status::fail(Reason::PipelineInvalid) : consumed;
  }
  if (!wavefront.release(ticket.batch)) {
    child_poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  return reset_ticket(ticket) ? Status::success()
                              : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::graph_reduce
