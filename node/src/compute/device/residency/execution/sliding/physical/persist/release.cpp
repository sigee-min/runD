#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::release_execution_sliding_persist(
    execution::Sliding &sliding, execution::SlidingPersist &&ticket) noexcept {
  if (sliding.state_ == nullptr || !ticket) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::Sliding::State::OutputCell *const output =
      state.output(ticket.ticket_, execution::SlidingTicketKind::Persist);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || output == nullptr ||
      output->state != execution::OutputState::Completing ||
      !output->physical_handoff || ticket.nonce_ != output->turn ||
      ticket.frame_ != output->physical_frame ||
      ticket.frame_ >= authority_.frames_.size()) {
    return false;
  }
  const bool known =
      output->completion_terminal == execution::TerminalKind::Known;
  if (!known) {
    output->state = execution::OutputState::Quarantined;
  } else {
    registry_model::Frame &host = authority_.frames_[ticket.frame_];
    if (output->completion && state.accepts(output->coordinate) &&
        !output->invalidate_after_terminal) {
      host.state = FrameState::Resident;
      host.dirty = {};
    }
    if (state.has_failure &&
        output->coordinate.ordinal >= state.first_failure.ordinal &&
        output->completion) {
      ++state.persist_completed_after_failure;
      static_cast<void>(execution::add(state.persist_bytes_after_failure,
                                       output->expected_bytes));
    }
    output->state = execution::OutputState::Persisted;
    state.advance_persist_frontier();
  }
  output->physical_handoff = false;
  ticket.ticket_ = {};
  ticket.key_ = {};
  ticket.extent_ = {};
  ticket.frame_ = 0u;
  ticket.nonce_ = 0u;
  return true;
}

} // namespace rund::compute::detail::residency
