#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::reuse_fetch(const SlidingProjection &projection,
                          const std::span<residency::PageUse> uses,
                          const std::size_t use) noexcept {
  if (state_ == nullptr || use > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  SlidingProjection expected{};
  std::uint64_t expected_bytes = 0u;
  if (!state_->model_only || !state_->runnable() ||
      !state_->accepts(projection.coordinate) ||
      projection.coordinate.ordinal !=
          state_->bank_frontier(projection.coordinate) ||
      projection.coordinate.ordinal >= state_->planned ||
      !state_->exact(projection, uses, expected) ||
      !state_->invocation.fetch_use(projection, uses, use) ||
      !state_->invocation.use_bytes(projection, uses, use, expected_bytes)) {
    return false;
  }
  const std::span<State::InputCell> ring =
      state_->input_ring(projection.coordinate);
  if (std::any_of(ring.begin(), ring.end(), [&](const State::InputCell &cell) {
        return cell.state != InputState::Free &&
               cell.coordinate == projection.coordinate && cell.use == use;
      })) {
    return false;
  }
  const residency::PageUse &demand = uses[use];
  const auto found =
      std::find_if(ring.begin(), ring.end(), [&](const State::InputCell &cell) {
        return cell.state == InputState::Ready && cell.key == demand.key &&
               cell.expected_bytes == expected_bytes &&
               cell.coordinate != projection.coordinate;
      });
  if (found == ring.end() || !add(state_->fetch_hits, 1u)) {
    return false;
  }
  found->coordinate = projection.coordinate;
  found->turn = found->turn == std::numeric_limits<std::uint64_t>::max()
                    ? 1u
                    : found->turn + 1u;
  found->use = static_cast<std::uint32_t>(use);
  found->next_use = demand.next_use;
  found->pin = demand.pin;
  return true;
}

bool Sliding::issue_fetch(const SlidingProjection &projection,
                          const std::span<residency::PageUse> uses,
                          const std::size_t use,
                          SlidingTicket &ticket) noexcept {
  ticket = {};
  if (state_ == nullptr || use > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  SlidingProjection expected{};
  std::uint64_t expected_bytes = 0u;
  if (!state_->model_only || !state_->runnable() ||
      !state_->accepts(projection.coordinate) ||
      projection.coordinate.ordinal < state_->admitted ||
      projection.coordinate.ordinal >= state_->planned ||
      !state_->exact(projection, uses, expected) ||
      !state_->invocation.fetch_use(projection, uses, use) ||
      !state_->invocation.use_bytes(projection, uses, use, expected_bytes)) {
    return false;
  }
  const std::span<State::InputCell> ring =
      state_->input_ring(projection.coordinate);
  if (std::any_of(ring.begin(), ring.end(), [&](const State::InputCell &cell) {
        return cell.state != InputState::Free &&
               cell.coordinate == projection.coordinate && cell.use == use;
      })) {
    return false;
  }
  const residency::PageUse &demand = uses[use];
  const std::uint64_t bank_frontier =
      state_->bank_frontier(projection.coordinate);
  const auto eligible = [&](const State::InputCell &cell) noexcept {
    return cell.state == InputState::Ready &&
           cell.coordinate.ordinal != bank_frontier &&
           cell.pin.last_epoch < projection.coordinate.ordinal &&
           (projection.coordinate.ordinal == bank_frontier ||
            cell.next_use > projection.coordinate.ordinal);
  };
  if (projection.coordinate.ordinal != bank_frontier) {
    const std::size_t current = static_cast<std::size_t>(std::count_if(
        ring.begin(), ring.end(), [&](const State::InputCell &cell) {
          return cell.state != InputState::Free &&
                 cell.coordinate.ordinal == bank_frontier;
        }));
    const std::size_t available = static_cast<std::size_t>(std::count_if(
        ring.begin(), ring.end(), [&](const State::InputCell &cell) {
          return cell.state == InputState::Free || eligible(cell);
        }));
    const std::size_t reserved = state_->coordinate_input_count > current
                                     ? state_->coordinate_input_count - current
                                     : 0u;
    if (available <= reserved) {
      return false;
    }
  }
  auto found =
      std::find_if(ring.begin(), ring.end(), [](const State::InputCell &cell) {
        return cell.state == InputState::Free;
      });
  if (found == ring.end()) {
    // Deterministic bounded Belady scan. The incoming page participates in
    // the comparison, so a farther-use forecast defers instead of evicting a
    // nearer resident. Equal next-use keeps the lower physical slot.
    for (auto candidate = ring.begin(); candidate != ring.end(); ++candidate) {
      if (eligible(*candidate) &&
          (found == ring.end() || candidate->next_use > found->next_use)) {
        found = candidate;
      }
    }
  }
  if (found == ring.end()) {
    return false;
  }
  const std::uint32_t slot = static_cast<std::uint32_t>(found - ring.begin());
  found->coordinate = projection.coordinate;
  found->key = demand.key;
  found->pin = demand.pin;
  found->turn = found->turn == std::numeric_limits<std::uint64_t>::max()
                    ? 1u
                    : found->turn + 1u;
  found->expected_bytes = expected_bytes;
  found->next_use = demand.next_use;
  found->use = static_cast<std::uint32_t>(use);
  found->state = InputState::Fetching;
  if (!add(state_->fetch_calls, 1u)) {
    found->state = InputState::Free;
    return false;
  }
  ticket = SlidingTicket{.plan = state_->plan,
                         .coordinate = projection.coordinate,
                         .token = state_->token,
                         .generation = state_->generation,
                         .owner = state_->owner,
                         .turn = found->turn,
                         .expected_bytes = expected_bytes,
                         .slot = slot,
                         .use = static_cast<std::uint32_t>(use),
                         .kind = SlidingTicketKind::Fetch};
  return true;
}

bool Sliding::fetch_terminal(const SlidingTicket &ticket, const Status status,
                             const TerminalKind terminal, const bool may_write,
                             const std::uint64_t bytes) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::InputCell *const cell =
      state_->input(ticket, SlidingTicketKind::Fetch);
  if (!state_->model_only || !state_->runnable() || cell == nullptr ||
      cell->state != InputState::Fetching) {
    return false;
  }
  if (terminal == TerminalKind::UnknownMayWrite) {
    state_->fail(ticket.coordinate, status ? integrity_failure() : status,
                 terminal, true);
    cell->state = InputState::Quarantined;
    return true;
  }
  const bool valid_bytes =
      status ? bytes == ticket.expected_bytes : bytes <= ticket.expected_bytes;
  if (!valid_bytes || !add(state_->fetch_bytes, bytes)) {
    state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                 true);
    cell->state = InputState::Invalidating;
    return true;
  }
  if (status) {
    cell->state = state_->accepts(ticket.coordinate) ? InputState::Ready
                                                     : InputState::Free;
    return true;
  }
  state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
  cell->state = may_write ? InputState::Invalidating : InputState::Free;
  return true;
}

} // namespace rund::compute::detail::residency::execution
