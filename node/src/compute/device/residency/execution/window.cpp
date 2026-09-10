#include "window.hpp"

#include "../registry/execution_owner.hpp"

#include <algorithm>

namespace rund::compute::detail::residency::execution {

ExecutionLease Window::begin(Authority &authority, const Plan &plan) noexcept {
  if (authority_ != nullptr || plan.epoch_count() == 0u ||
      plan.epoch_count() > WindowCapacity) {
    return ExecutionLease{.failure = AuthorityFailure::Invalid};
  }
  ExecutionLease lease = authority.executions().begin_execution_window(plan);
  if (!lease || !service_.bind(authority, plan, lease)) {
    return lease ? ExecutionLease{.failure = AuthorityFailure::Invalid} : lease;
  }
  authority_ = &authority;
  plan_ = &plan;
  lease_ = lease;
  return lease_;
}

bool Window::issue(const std::uint64_t epoch, const Phase phase,
                   ExecutionTicket &ticket) noexcept {
  ticket = {};
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      epoch >= lease_.epochs ||
      (phase != Phase::Input && phase != Phase::Output)) {
    return false;
  }
  const std::size_t index = static_cast<std::size_t>(epoch);
  if ((phase == Phase::Input && (aborted_ || input_issued_[index])) ||
      (phase == Phase::Output &&
       (output_issued_[index] || index >= release_count_ ||
        !releases_[index].status || !releases_[index].completed))) {
    return false;
  }
  if (!service_.issue(epoch, phase, ticket)) {
    return false;
  }
  (phase == Phase::Input ? input_issued_[index] : output_issued_[index]) = true;
  return true;
}

bool Window::terminal(const ExecutionTicket &ticket,
                      const ExecutionTerminal terminal,
                      const Status status) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      ticket.epoch >= lease_.epochs ||
      (ticket.phase != static_cast<std::uint8_t>(Phase::Input) &&
       ticket.phase != static_cast<std::uint8_t>(Phase::Output))) {
    return false;
  }
  const std::size_t index = static_cast<std::size_t>(ticket.epoch);
  const bool input = ticket.phase == static_cast<std::uint8_t>(Phase::Input);
  if ((input && (!input_issued_[index] || input_done_[index])) ||
      (!input && (!output_issued_[index] || output_done_[index])) ||
      !service_.terminal(ticket, terminal, status)) {
    return false;
  }
  (input ? input_done_[index] : output_done_[index]) = true;
  if (terminal != ExecutionTerminal::Success) {
    aborted_ = true;
    unknown_ = terminal == ExecutionTerminal::UnknownMayWrite;
  }
  advance_ready();
  return true;
}

bool Window::release(const Release &release) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      release_count_ >= lease_.epochs || release.epoch != release_count_ ||
      (release.dispatched && (release.status || release.may_write) &&
       !(unknown_ && release.terminal == TerminalKind::UnknownMayWrite) &&
       !input_done_[static_cast<std::size_t>(release.epoch)]) ||
      !service_.release(release)) {
    return false;
  }
  releases_[release_count_++] = release;
  native_batches_ += release.dispatched ? 1u : 0u;
  if (release.status && release_count_ - 1u == release_prefix_) {
    ++release_prefix_;
  } else {
    aborted_ = true;
    unknown_ = unknown_ || release.terminal == TerminalKind::UnknownMayWrite;
  }
  advance_ready();
  return true;
}

bool Window::final(const WindowEvidence &evidence) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      evidence.release_count != release_count_ ||
      evidence.native_batches != native_batches_ ||
      !authority_->executions().accept_execution_window(*plan_, evidence)) {
    return false;
  }
  queue_calls_ = evidence.queue_calls;
  final_ = true;
  return true;
}

bool Window::abandon_final(const WindowEvidence &evidence) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      evidence.release_count != lease_.epochs ||
      evidence.terminal != TerminalKind::Known ||
      !authority_->executions().abandon_execution_window(*plan_, evidence)) {
    return false;
  }
  aborted_ = true;
  closed_ = true;
  return true;
}

ExecutionClose Window::close() noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || !final_) {
    return ExecutionClose{.failure = !final_ ? AuthorityFailure::Busy
                                             : AuthorityFailure::Invalid};
  }
  ExecutionClose result =
      authority_->executions().close_execution(lease_.token, lease_.generation, *plan_);
  closed_ = static_cast<bool>(result);
  return result;
}

bool Window::abandon() noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      release_count_ != 0u || native_batches_ != 0u ||
      std::any_of(output_issued_.begin(), output_issued_.end(),
                  [](const bool issued) { return issued; })) {
    return false;
  }
  const bool abandoned =
      authority_->executions().abandon_execution(lease_.token, lease_.generation, *plan_);
  closed_ = abandoned;
  return abandoned;
}

WindowSnapshot Window::snapshot() const noexcept {
  return WindowSnapshot{.release_prefix = release_prefix_,
                        .ready_prefix = ready_prefix_,
                        .native_batches = native_batches_,
                        .queue_calls = queue_calls_,
                        .aborted = aborted_,
                        .unknown = unknown_,
                        .final = final_};
}

void Window::advance_ready() noexcept {
  while (ready_prefix_ < lease_.epochs) {
    const std::size_t index = static_cast<std::size_t>(ready_prefix_);
    if (!input_done_[index] || index >= release_count_ ||
        !releases_[index].status || !releases_[index].completed ||
        !output_done_[index]) {
      return;
    }
    ++ready_prefix_;
  }
}

} // namespace rund::compute::detail::residency::execution
