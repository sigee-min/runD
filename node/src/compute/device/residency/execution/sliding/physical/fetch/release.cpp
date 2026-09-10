#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::release_execution_sliding_fetch(
    execution::Sliding &sliding, execution::SlidingFetch &&ticket) noexcept {
  if (sliding.state_ == nullptr || !ticket) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::Sliding::State::InputCell *const cell =
      state.input(ticket.ticket_, execution::SlidingTicketKind::Fetch);
  registry_model::Frame *reuse_frame = nullptr;
  if (ticket.reuses_frame_) {
    if (!ticket.reuse_ || ticket.reuse_frame_ >= authority_.frames_.size()) {
      return false;
    }
    reuse_frame = &authority_.frames_[ticket.reuse_frame_];
    if (reuse_frame->state != FrameState::Pinned ||
        reuse_frame->key != ticket.reuse_.key) {
      return false;
    }
  }
  const std::uint32_t target_frame = ticket.frame_;
  const CacheKey target_key = ticket.source_.key;
  const std::uint32_t source_frame = ticket.reuse_frame_;
  const CacheKey source_key = ticket.reuse_.key;
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || cell == nullptr ||
      cell->physical_frame != ticket.frame_ || ticket.nonce_ != cell->turn ||
      ticket.frame_ >= authority_.frames_.size() || !cell->physical_handoff) {
    return false;
  }
  if (!ticket.backing_) {
    if (cell->state != execution::InputState::Ready ||
        authority_.frames_[ticket.frame_].state != FrameState::Resident) {
      return false;
    }
    cell->physical_handoff = false;
    if (reuse_frame != nullptr) {
      reuse_frame->state = FrameState::Resident;
      observe_sliding_frame(source_frame, source_key);
    }
    observe_sliding_frame(target_frame, target_key);
    if (!state.accepts(cell->coordinate)) {
      cell->state = execution::InputState::Free;
    }
    ticket.ticket_ = {};
    ticket.source_ = {};
    ticket.reuse_ = {};
    ticket.frame_ = 0u;
    ticket.reuse_frame_ = 0u;
    ticket.backing_bytes_ = 0u;
    ticket.nonce_ = 0u;
    ticket.backing_ = false;
    ticket.reuses_frame_ = false;
    return true;
  }
  if (cell->state != execution::InputState::Completing) {
    return false;
  }
  registry_model::Frame &frame = authority_.frames_[ticket.frame_];
  if (cell->completion_terminal == execution::TerminalKind::UnknownMayWrite) {
    cell->state = execution::InputState::Quarantined;
    clear_sliding_frame_observation(target_frame);
  } else if (cell->completion) {
    frame.state = FrameState::Resident;
    observe_sliding_frame(target_frame, target_key);
    cell->state = state.accepts(cell->coordinate) ? execution::InputState::Ready
                                                  : execution::InputState::Free;
  } else {
    clear_sliding_frame_observation(target_frame);
    cell->state = cell->completion_may_write
                      ? execution::InputState::Invalidating
                      : execution::InputState::Free;
  }
  cell->physical_handoff = false;
  if (reuse_frame != nullptr) {
    reuse_frame->state = FrameState::Resident;
    if (cell->completion) {
      observe_sliding_frame(source_frame, source_key);
    }
  }
  ticket.ticket_ = {};
  ticket.source_ = {};
  ticket.reuse_ = {};
  ticket.frame_ = 0u;
  ticket.reuse_frame_ = 0u;
  ticket.backing_bytes_ = 0u;
  ticket.nonce_ = 0u;
  ticket.backing_ = false;
  ticket.reuses_frame_ = false;
  return true;
}

} // namespace rund::compute::detail::residency
