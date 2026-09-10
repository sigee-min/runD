#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::terminal_execution_sliding_drain(
    execution::Sliding &sliding, const execution::SlidingDrain &ticket,
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
  execution::Sliding::State::OutputCell *const output =
      state.output(ticket.ticket_, execution::SlidingTicketKind::Drain);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || output == nullptr ||
      output->state != execution::OutputState::Draining ||
      !output->physical_handoff || ticket.nonce_ != output->turn ||
      ticket.host_frame_ != output->physical_frame) {
    return false;
  }
  execution::Sliding::State::NativeCell &native =
      state.native_cells[static_cast<std::size_t>(
          ticket.ticket_.coordinate.ordinal %
          execution::SlidingNativeCapacity)];
  if (native.coordinate != ticket.ticket_.coordinate ||
      native.state != execution::NativeState::Draining ||
      std::find(native.device_output_frames.begin(),
                native.device_output_frames.begin() + native.output_count,
                ticket.device_frame_) ==
          native.device_output_frames.begin() + native.output_count) {
    return false;
  }
  const bool known = terminal == execution::TerminalKind::Known;
  const bool valid_bytes = status ? bytes == ticket.ticket_.expected_bytes
                                  : bytes <= ticket.ticket_.expected_bytes;
  const bool integrity = !valid_bytes || (!known && status);
  output->completion =
      integrity ? Status::fail(Reason::CompletionInvalid) : status;
  output->completion_terminal = terminal;
  output->completion_bytes = bytes;
  output->completion_may_write =
      may_write || integrity || !known || bytes != 0u;
  if (known && valid_bytes &&
      bytes <= std::numeric_limits<std::uint64_t>::max() - state.drain_bytes) {
    state.drain_bytes += bytes;
  } else if (known && valid_bytes) {
    output->completion = Status::fail(Reason::CompletionInvalid);
    output->completion_may_write = true;
  }
  if (!output->completion || !known) {
    state.fail(output->coordinate, output->completion, terminal,
               output->completion_may_write);
  }
  output->state = execution::OutputState::Completing;
  return true;
}

} // namespace rund::compute::detail::residency
