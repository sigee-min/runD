#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::terminal_execution_sliding_native(
    execution::Sliding &sliding, const execution::SlidingNative &ticket,
    const Status status, const execution::TerminalKind terminal,
    const bool may_write) noexcept {
  if (sliding.state_ == nullptr || !ticket) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::Sliding::State::NativeCell *const native =
      state.native(ticket.ticket_, execution::SlidingTicketKind::Native);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || native == nullptr ||
      native->state != execution::NativeState::Submitted ||
      !native->physical_handoff || ticket.nonce_ != native->turn ||
      ticket.input_count_ != native->input_count ||
      ticket.output_count_ != native->output_count ||
      ticket.input_frames_ != native->device_input_frames ||
      ticket.output_frames_ != native->device_output_frames) {
    return false;
  }
  const bool known = terminal == execution::TerminalKind::Known;
  const bool integrity = !known && status;
  native->completion =
      integrity ? Status::fail(Reason::CompletionInvalid) : status;
  native->completion_terminal = terminal;
  native->completion_may_write = may_write || integrity || !known ||
                                 (status && native->output_count != 0u);
  if (!native->completion || !known) {
    state.fail(native->coordinate, native->completion, terminal,
               native->completion_may_write);
  }
  native->state = execution::NativeState::Completing;
  return true;
}

} // namespace rund::compute::detail::residency
