#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

bool Sliding::issue_persist(const SlidingTicket &drain_ticket,
                            SlidingTicket &ticket) noexcept {
  ticket = {};
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::OutputCell *const output =
      state_->output(drain_ticket, SlidingTicketKind::Drain);
  if (!state_->model_only || !state_->runnable() || output == nullptr ||
      output->state != OutputState::Ready ||
      !state_->accepts(drain_ticket.coordinate)) {
    return false;
  }
  const auto earlier = [&](const State::OutputCell &cell) {
    if (cell.state == OutputState::Free || &cell == output) {
      return false;
    }
    return cell.coordinate.ordinal < output->coordinate.ordinal ||
           (cell.coordinate.ordinal == output->coordinate.ordinal &&
            cell.use < output->use);
  };
  if (std::any_of(state_->outputs.begin(),
                  state_->outputs.begin() + state_->output_count, earlier)) {
    return false;
  }
  output->state = OutputState::Persisting;
  output->persist_sequence = state_->persist_issued++;
  if (!add(state_->persist_calls, 1u)) {
    --state_->persist_issued;
    output->persist_sequence = NeverUse;
    output->state = OutputState::Ready;
    return false;
  }
  ticket = drain_ticket;
  ticket.kind = SlidingTicketKind::Persist;
  return true;
}

bool Sliding::persist_terminal(const SlidingTicket &ticket, const Status status,
                               const TerminalKind terminal,
                               const bool may_write,
                               const std::uint64_t bytes) noexcept {
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  State::OutputCell *const output =
      state_->output(ticket, SlidingTicketKind::Persist);
  if (!state_->model_only || !state_->runnable() || output == nullptr ||
      output->state != OutputState::Persisting) {
    return false;
  }
  if (terminal == TerminalKind::UnknownMayWrite) {
    state_->fail(ticket.coordinate, status ? integrity_failure() : status,
                 terminal, true);
    output->state = OutputState::Quarantined;
    return true;
  }
  const bool valid_bytes =
      status ? bytes == ticket.expected_bytes : bytes <= ticket.expected_bytes;
  if (!valid_bytes || !add(state_->persist_bytes, bytes)) {
    state_->fail(ticket.coordinate, integrity_failure(), TerminalKind::Known,
                 true);
    output->state = OutputState::Invalidating;
    return true;
  }
  if (!status) {
    state_->fail(ticket.coordinate, status, TerminalKind::Known, may_write);
    output->state = may_write ? OutputState::Invalidating : OutputState::Free;
    if (!may_write && output->persist_sequence == state_->persist_frontier) {
      ++state_->persist_frontier;
      state_->advance_persist_frontier();
    }
    return true;
  }
  if (state_->has_failure &&
      output->coordinate.ordinal >= state_->first_failure.ordinal) {
    ++state_->persist_completed_after_failure;
    static_cast<void>(
        add(state_->persist_bytes_after_failure, output->expected_bytes));
  }
  output->state = output->invalidate_after_terminal ? OutputState::Invalidating
                                                    : OutputState::Persisted;
  state_->advance_persist_frontier();
  return true;
}

} // namespace rund::compute::detail::residency::execution
