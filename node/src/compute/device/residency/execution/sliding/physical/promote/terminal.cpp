#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::terminal_execution_sliding_promote(
    execution::Sliding &sliding, const execution::SlidingPromote &ticket,
    const Status status, const execution::TerminalKind terminal,
    const bool may_write, const std::uint64_t bytes) noexcept {
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
      state.native(ticket.ticket_, execution::SlidingTicketKind::Promote);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || native == nullptr ||
      native->state != execution::NativeState::Promoting ||
      !native->physical_handoff || ticket.nonce_ != native->turn ||
      ticket.source_count_ != native->source_count ||
      ticket.input_count_ != native->input_count ||
      ticket.output_count_ != native->output_count ||
      ticket.transfer_mask_ != native->transfer_mask ||
      ticket.host_frames_ != native->host_frames ||
      ticket.device_input_frames_ != native->device_input_frames ||
      ticket.device_output_frames_ != native->device_output_frames) {
    return false;
  }
  const bool known = terminal == execution::TerminalKind::Known;
  const bool valid_bytes = status ? bytes == ticket.ticket_.expected_bytes
                                  : bytes <= ticket.ticket_.expected_bytes;
  const bool integrity = !valid_bytes || (!known && status);
  const bool effective_may_write = may_write || integrity || !known ||
                                   bytes != 0u ||
                                   (status && ticket.transfer_mask_ != 0u);
  const Status completion =
      integrity ? Status::fail(Reason::CompletionInvalid) : status;
  if (known && valid_bytes &&
      bytes > std::numeric_limits<std::uint64_t>::max() - state.promote_bytes) {
    native->completion = Status::fail(Reason::CompletionInvalid);
    native->completion_terminal = execution::TerminalKind::Known;
    native->completion_may_write = true;
    native->completion_bytes = 0u;
    state.fail(ticket.ticket_.coordinate,
               Status::fail(Reason::CompletionInvalid),
               execution::TerminalKind::Known, true);
  } else {
    native->completion = completion;
    native->completion_terminal = terminal;
    native->completion_may_write = effective_may_write;
    native->completion_bytes = bytes;
    if (known && valid_bytes) {
      state.promote_bytes += bytes;
    }
    if (!completion || !known) {
      state.fail(ticket.ticket_.coordinate, completion, terminal,
                 effective_may_write);
    }
  }
  native->state = execution::NativeState::Completing;
  return true;
}

} // namespace rund::compute::detail::residency
