#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::invalidate(const SlidingTicket &ticket) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  if (!state_->model_only || !state_->runnable() ||
      !state_->credential(ticket)) {
    return false;
  }
  if (ticket.kind == SlidingTicketKind::Fetch) {
    State::InputCell *const input =
        state_->input(ticket, SlidingTicketKind::Fetch);
    if (input == nullptr || input->state != InputState::Invalidating) {
      return false;
    }
    input->state = InputState::Free;
    return true;
  }
  if (ticket.kind == SlidingTicketKind::Drain ||
      ticket.kind == SlidingTicketKind::Persist) {
    State::OutputCell *const output = state_->output(ticket, ticket.kind);
    if (output == nullptr || output->state != OutputState::Invalidating) {
      return false;
    }
    if (output->persist_sequence == state_->persist_frontier) {
      ++state_->persist_frontier;
      state_->advance_persist_frontier();
    }
    output->state = OutputState::Free;
    State::NativeCell &native = state_->native_cells[static_cast<std::size_t>(
        ticket.coordinate.ordinal % SlidingNativeCapacity)];
    if (native.coordinate == ticket.coordinate &&
        native.state == NativeState::Draining) {
      ++native.resolved;
      state_->resolve_native(native);
    }
    return true;
  }
  State::NativeCell *const native = state_->native(ticket, ticket.kind);
  if (native == nullptr) {
    return false;
  }
  State::TerminalCell &terminal = state_->terminals[static_cast<std::size_t>(
      ticket.coordinate.ordinal % SlidingNativeCapacity)];
  if (terminal.ordinal == ticket.coordinate.ordinal && terminal.invalidating) {
    terminal.invalidating = false;
    terminal.terminal = true;
    state_->advance_frontier();
    return true;
  }
  if (native->state != NativeState::Invalidating) {
    return false;
  }
  for (State::InputCell &input : state_->inputs) {
    if (input.coordinate == ticket.coordinate &&
        input.state == InputState::Promoting) {
      input.state = InputState::Free;
    }
  }
  for (State::OutputCell &output : state_->outputs) {
    if (output.coordinate == ticket.coordinate &&
        (output.state == OutputState::Invalidating ||
         output.state == OutputState::Reserved)) {
      output.state = OutputState::Free;
    }
  }
  const bool dispatched = ticket.kind == SlidingTicketKind::Native;
  native->state = NativeState::Free;
  if (dispatched) {
    state_->retire(ticket.coordinate, false);
  }
  return true;
}

} // namespace rund::compute::detail::residency::execution
