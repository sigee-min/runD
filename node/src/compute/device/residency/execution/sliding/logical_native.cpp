#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::admit(const SlidingProjection &projection,
                    const std::span<residency::PageUse> uses,
                    SlidingTicket &ticket) noexcept {
  ticket = {};
  if (state_ == nullptr || projection.use_count > uses.size()) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  SlidingProjection expected{};
  State::NativeCell &native = state_->native_cells[static_cast<std::size_t>(
      projection.coordinate.ordinal % SlidingNativeCapacity)];
  if (!state_->model_only || !state_->runnable() ||
      !state_->accepts(projection.coordinate) ||
      !state_->exact(projection, uses, expected) ||
      projection.coordinate.ordinal != state_->admitted ||
      native.coordinate != projection.coordinate ||
      native.state != NativeState::Ready) {
    return false;
  }
  native.state = NativeState::Submitted;
  ++state_->admitted;
  ticket = SlidingTicket{
      .plan = state_->plan,
      .coordinate = projection.coordinate,
      .token = state_->token,
      .generation = state_->generation,
      .owner = state_->owner,
      .turn = native.turn,
      .expected_bytes = native.promote_bytes,
      .slot = static_cast<std::uint32_t>(projection.coordinate.ordinal %
                                         SlidingNativeCapacity),
      .kind = SlidingTicketKind::Native,
  };
  return true;
}

bool Sliding::native_terminal(const SlidingTicket &ticket, const Status status,
                              const TerminalKind terminal,
                              const bool may_write) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::NativeCell *const native =
      state_->native(ticket, SlidingTicketKind::Native);
  if (!state_->model_only || !state_->runnable() || native == nullptr ||
      native->state != NativeState::Submitted) {
    return false;
  }
  if (terminal == TerminalKind::UnknownMayWrite) {
    state_->fail(ticket.coordinate, status ? integrity_failure() : status,
                 terminal, true);
    native->state = NativeState::Quarantined;
    for (State::OutputCell &output : state_->outputs) {
      if (output.state != OutputState::Free &&
          output.coordinate == ticket.coordinate) {
        output.state = OutputState::Quarantined;
      }
    }
    for (State::InputCell &input : state_->inputs) {
      if (input.state == InputState::Promoting &&
          input.coordinate == ticket.coordinate) {
        input.state = InputState::Quarantined;
      }
    }
    return true;
  }
  for (State::InputCell &input : state_->inputs) {
    if (input.state == InputState::Promoting &&
        input.coordinate == ticket.coordinate) {
      input.state = InputState::Ready;
    }
  }
  if (native->suppress && status) {
    // fail() observed this suffix while its Host rows were still Promoting.
    // They therefore were not part of fail()'s Ready-row cancellation scan.
    // The authenticated no-write terminal must retire that speculative cache
    // metadata now; a failed suffix never seeds a later cache hit.
    for (State::InputCell &input : state_->inputs) {
      if (input.state == InputState::Ready &&
          input.coordinate == ticket.coordinate) {
        input.state = InputState::Free;
      }
    }
    if (may_write) {
      state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                   true);
      native->state = NativeState::Invalidating;
      for (State::OutputCell &output : state_->outputs) {
        if (output.state == OutputState::Reserved &&
            output.coordinate == ticket.coordinate) {
          output.state = OutputState::Invalidating;
        }
      }
    } else {
      state_->cancel_reserved(ticket.coordinate, *native);
      native->state = NativeState::Free;
      state_->retire(ticket.coordinate, false);
    }
    return true;
  }
  if (!status) {
    state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
    if (may_write) {
      native->state = NativeState::Invalidating;
      for (State::OutputCell &output : state_->outputs) {
        if (output.state == OutputState::Reserved &&
            output.coordinate == ticket.coordinate) {
          output.state = OutputState::Invalidating;
        }
      }
      return true;
    }
    state_->cancel_reserved(ticket.coordinate, *native);
    native->state = NativeState::Free;
    state_->retire(ticket.coordinate, false);
    return true;
  }
  native->state = NativeState::Draining;
  if (native->output_count == 0u) {
    native->state = NativeState::Free;
    state_->retire(ticket.coordinate, may_write);
  }
  return true;
}

} // namespace rund::compute::detail::residency::execution
