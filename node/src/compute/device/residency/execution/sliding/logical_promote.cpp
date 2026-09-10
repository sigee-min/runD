#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::issue_promote(const SlidingProjection &projection,
                            const std::span<residency::PageUse> uses,
                            SlidingTicket &ticket) noexcept {
  ticket = {};
  if (state_ == nullptr || projection.use_count > uses.size()) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  SlidingProjection expected{};
  std::array<SlidingCoordinate, SlidingPredecessorCapacity> predecessors{};
  std::size_t predecessor_count = 0u;
  if (!state_->model_only || !state_->runnable() ||
      !state_->accepts(projection.coordinate) ||
      !state_->exact(projection, uses, expected) ||
      projection.coordinate.ordinal != state_->admitted ||
      state_->admitted - state_->terminal_frontier >= SlidingNativeCapacity ||
      !state_->invocation.predecessors(projection, predecessors,
                                       predecessor_count)) {
    return false;
  }
  for (std::size_t index = 0u; index < predecessor_count; ++index) {
    if (!state_->terminal_for(predecessors[index])) {
      return false;
    }
  }
  const std::span<State::InputCell> input_ring =
      state_->input_ring(projection.coordinate);
  const std::span<State::OutputCell> output_ring =
      state_->output_ring(projection.coordinate);
  std::size_t ready = 0u;
  for (std::size_t use = 0u; use < projection.use_count; ++use) {
    if (!state_->invocation.fetch_use(projection, uses, use)) {
      continue;
    }
    const auto found = std::find_if(
        input_ring.begin(), input_ring.end(),
        [&](const State::InputCell &cell) {
          return cell.state == InputState::Ready && !cell.physical_handoff &&
                 cell.coordinate == projection.coordinate && cell.use == use;
        });
    if (found == input_ring.end()) {
      return false;
    }
    ++ready;
  }
  const std::size_t free_outputs = static_cast<std::size_t>(
      std::count_if(output_ring.begin(), output_ring.end(),
                    [](const State::OutputCell &cell) {
                      return cell.state == OutputState::Free;
                    }));
  State::NativeCell &native = state_->native_cells[static_cast<std::size_t>(
      projection.coordinate.ordinal % SlidingNativeCapacity)];
  if (ready != projection.fetch_count ||
      projection.persist_count > free_outputs ||
      native.state != NativeState::Free) {
    return false;
  }

  for (std::size_t use = 0u; use < projection.use_count; ++use) {
    if (!state_->invocation.persist_use(projection, uses, use)) {
      continue;
    }
    std::uint64_t expected_bytes = 0u;
    const auto found = std::find_if(output_ring.begin(), output_ring.end(),
                                    [](const State::OutputCell &cell) {
                                      return cell.state == OutputState::Free;
                                    });
    if (found == output_ring.end() ||
        !state_->invocation.use_bytes(projection, uses, use, expected_bytes)) {
      return false;
    }
    found->coordinate = projection.coordinate;
    found->turn = found->turn == std::numeric_limits<std::uint64_t>::max()
                      ? 1u
                      : found->turn + 1u;
    found->expected_bytes = expected_bytes;
    found->use = static_cast<std::uint32_t>(use);
    found->state = OutputState::Reserved;
    found->invalidate_after_terminal = false;
  }
  for (State::InputCell &input : state_->inputs) {
    if (input.state == InputState::Ready &&
        input.coordinate == projection.coordinate) {
      input.state = InputState::Promoting;
    }
  }
  native.coordinate = projection.coordinate;
  native.turn = native.turn == std::numeric_limits<std::uint64_t>::max()
                    ? 1u
                    : native.turn + 1u;
  native.promote_bytes = projection.fetch_bytes;
  native.output_count = static_cast<std::uint32_t>(projection.persist_count);
  native.resolved = 0u;
  native.state = NativeState::Promoting;
  native.suppress = false;
  native.invalidate_after_terminal = false;
  if (!add(state_->promote_calls, 1u)) {
    for (State::InputCell &input : state_->inputs) {
      if (input.state == InputState::Promoting &&
          input.coordinate == projection.coordinate) {
        input.state = InputState::Ready;
      }
    }
    state_->cancel_reserved(projection.coordinate, native);
    native.state = NativeState::Free;
    return false;
  }
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
      .kind = SlidingTicketKind::Promote,
  };
  return true;
}

bool Sliding::promote_terminal(const SlidingTicket &ticket, const Status status,
                               const TerminalKind terminal,
                               const bool may_write,
                               const std::uint64_t bytes) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::NativeCell *const native =
      state_->native(ticket, SlidingTicketKind::Promote);
  if (!state_->model_only || !state_->runnable() || native == nullptr ||
      native->state != NativeState::Promoting) {
    return false;
  }
  if (terminal == TerminalKind::UnknownMayWrite) {
    state_->fail(ticket.coordinate, status ? integrity_failure() : status,
                 terminal, true);
    native->state = NativeState::Quarantined;
    for (State::InputCell &cell : state_->inputs) {
      if (cell.state == InputState::Promoting &&
          cell.coordinate == ticket.coordinate) {
        cell.state = InputState::Quarantined;
      }
    }
    for (State::OutputCell &cell : state_->outputs) {
      if (cell.state == OutputState::Reserved &&
          cell.coordinate == ticket.coordinate) {
        cell.state = OutputState::Quarantined;
      }
    }
    return true;
  }
  const bool valid_bytes =
      status ? bytes == ticket.expected_bytes : bytes <= ticket.expected_bytes;
  if (!valid_bytes || !add(state_->promote_bytes, bytes)) {
    state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                 true);
    for (State::InputCell &cell : state_->inputs) {
      if (cell.state == InputState::Promoting &&
          cell.coordinate == ticket.coordinate) {
        cell.state = InputState::Free;
      }
    }
    state_->cancel_reserved(ticket.coordinate, *native);
    native->state = NativeState::Invalidating;
    return true;
  }
  if (!status || !state_->accepts(ticket.coordinate)) {
    if (!status) {
      state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
    }
    for (State::InputCell &cell : state_->inputs) {
      if (cell.state == InputState::Promoting &&
          cell.coordinate == ticket.coordinate) {
        cell.state = InputState::Free;
      }
    }
    state_->cancel_reserved(ticket.coordinate, *native);
    native->state = may_write ? NativeState::Invalidating : NativeState::Free;
    return true;
  }
  // Keep the Host row pinned through the native terminal. A coherent lowering
  // may alias this exact Host storage instead of copying it, so a Promote
  // receipt alone is not sufficient authority to overwrite the bytes.
  native->state = NativeState::Ready;
  return true;
}

} // namespace rund::compute::detail::residency::execution
