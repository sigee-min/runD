#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::terminal_execution_sliding_fetch(
    execution::Sliding &sliding, const execution::SlidingFetch &ticket,
    const Status status, const execution::TerminalKind terminal,
    const bool may_write, const std::uint64_t bytes) noexcept {
  if (sliding.state_ == nullptr || !ticket || !ticket.backing_) {
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
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || cell == nullptr ||
      cell->state != execution::InputState::Fetching ||
      cell->physical_frame != ticket.frame_ || ticket.nonce_ != cell->turn ||
      ticket.frame_ >= authority_.frames_.size() ||
      authority_.frames_[ticket.frame_].state != FrameState::Mapping) {
    return false;
  }
  const bool known = terminal == execution::TerminalKind::Known;
  const bool valid_bytes =
      status ? bytes == ticket.backing_bytes_ : bytes <= ticket.backing_bytes_;
  const bool integrity = !valid_bytes || (!known && status);
  const bool effective_may_write = may_write || integrity || !known;
  const Status completion =
      integrity ? Status::fail(Reason::CompletionInvalid) : status;
  if (known && valid_bytes &&
      bytes > std::numeric_limits<std::uint64_t>::max() - state.fetch_bytes) {
    state.fail(ticket.ticket_.coordinate,
               Status::fail(Reason::CompletionInvalid),
               execution::TerminalKind::Known, true);
    cell->completion = Status::fail(Reason::CompletionInvalid);
    cell->completion_terminal = execution::TerminalKind::Known;
    cell->completion_may_write = true;
    cell->completion_bytes = 0u;
  } else {
    if (known && valid_bytes) {
      state.fetch_bytes += bytes;
    }
    cell->completion = completion;
    cell->completion_terminal = terminal;
    cell->completion_may_write = effective_may_write;
    cell->completion_bytes = bytes;
    if (!completion || !known) {
      state.fail(ticket.ticket_.coordinate, completion, terminal,
                 effective_may_write);
    }
  }
  cell->state = execution::InputState::Completing;
  return true;
}

} // namespace rund::compute::detail::residency
