#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::issue_drain(const SlidingTicket &native_ticket,
                          const std::size_t use,
                          SlidingTicket &ticket) noexcept {
  ticket = {};
  if (state_ == nullptr || use > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::NativeCell *const native =
      state_->native(native_ticket, SlidingTicketKind::Native);
  if (!state_->model_only || !state_->runnable() || native == nullptr ||
      native->state != NativeState::Draining) {
    return false;
  }
  const std::span<State::OutputCell> output_ring =
      state_->output_ring(native_ticket.coordinate);
  const auto found =
      std::find_if(output_ring.begin(), output_ring.end(),
                   [&](const State::OutputCell &output) {
                     return output.state == OutputState::Reserved &&
                            output.coordinate == native_ticket.coordinate &&
                            output.use == use;
                   });
  if (found == output_ring.end()) {
    return false;
  }
  found->state = OutputState::Draining;
  if (!add(state_->drain_calls, 1u)) {
    found->state = OutputState::Reserved;
    return false;
  }
  ticket = SlidingTicket{
      .plan = state_->plan,
      .coordinate = found->coordinate,
      .token = state_->token,
      .generation = state_->generation,
      .owner = state_->owner,
      .turn = found->turn,
      .expected_bytes = found->expected_bytes,
      .slot = static_cast<std::uint32_t>(found - output_ring.begin()),
      .use = found->use,
      .kind = SlidingTicketKind::Drain,
  };
  return true;
}

bool Sliding::drain_terminal(const SlidingTicket &ticket, const Status status,
                             const TerminalKind terminal, const bool may_write,
                             const std::uint64_t bytes) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::OutputCell *const output =
      state_->output(ticket, SlidingTicketKind::Drain);
  if (!state_->model_only || !state_->runnable() || output == nullptr ||
      output->state != OutputState::Draining) {
    return false;
  }
  State::NativeCell &native = state_->native_cells[static_cast<std::size_t>(
      ticket.coordinate.ordinal % SlidingNativeCapacity)];
  if (native.coordinate != ticket.coordinate ||
      (native.state != NativeState::Draining &&
       native.state != NativeState::Quarantined)) {
    return false;
  }
  if (terminal == TerminalKind::UnknownMayWrite) {
    state_->fail(ticket.coordinate, status ? integrity_failure() : status,
                 terminal, true);
    output->state = OutputState::Quarantined;
    native.state = NativeState::Quarantined;
    for (State::OutputCell &cell : state_->outputs) {
      if (cell.coordinate == ticket.coordinate &&
          cell.state == OutputState::Reserved) {
        // This Host slot was never exposed to a Drain operation and is safe
        // to cancel. Already-issued Draining siblings remain live until their
        // exact callbacks arrive through the Quarantined native owner above.
        cell.state = OutputState::Free;
      }
    }
    return true;
  }
  const bool valid_bytes =
      status ? bytes == ticket.expected_bytes : bytes <= ticket.expected_bytes;
  if (native.state == NativeState::Quarantined) {
    if (!valid_bytes || !add(state_->drain_bytes, bytes)) {
      state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                   true);
    } else if (!status) {
      state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
    }
    output->state = OutputState::Quarantined;
    return true;
  }
  if (!valid_bytes || !add(state_->drain_bytes, bytes)) {
    state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                 true);
    output->state = OutputState::Invalidating;
    native.invalidate_after_terminal = true;
    state_->cancel_reserved(ticket.coordinate, native);
    state_->resolve_native(native);
    return true;
  }
  if (!status) {
    state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
    output->state = may_write ? OutputState::Invalidating : OutputState::Free;
    if (!may_write) {
      ++native.resolved;
    }
    native.invalidate_after_terminal = true;
    for (State::OutputCell &cell : state_->outputs) {
      if (cell.coordinate == ticket.coordinate &&
          cell.state == OutputState::Ready) {
        cell.state = OutputState::Invalidating;
      } else if (cell.coordinate == ticket.coordinate &&
                 (cell.state == OutputState::Draining ||
                  cell.state == OutputState::Persisting)) {
        cell.invalidate_after_terminal = true;
      }
    }
    state_->cancel_reserved(ticket.coordinate, native);
    state_->resolve_native(native);
    return true;
  }
  if (native.invalidate_after_terminal || output->invalidate_after_terminal) {
    output->state = OutputState::Invalidating;
  } else {
    output->state = OutputState::Ready;
    ++native.resolved;
  }
  state_->resolve_native(native);
  return true;
}

} // namespace rund::compute::detail::residency::execution
