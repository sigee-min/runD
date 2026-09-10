#include "../../registry/execution_owner.hpp"
#include "../run.hpp"

namespace rund::compute::detail::residency::execution {

ExecutionLease Run::begin(Authority &authority, const Plan &plan) noexcept {
  if (authority_ != nullptr || plan.epoch_count() != 1u) {
    return ExecutionLease{.failure = AuthorityFailure::Invalid};
  }
  ExecutionLease lease = authority.executions().begin_execution(plan);
  if (!lease) {
    return lease;
  }
  authority_ = &authority;
  plan_ = &plan;
  lease_ = lease;
  return lease_;
}

bool Run::issue(const Phase phase, ExecutionTicket &ticket) noexcept {
  ticket = {};
  if (authority_ == nullptr || plan_ == nullptr || closed_ ||
      (phase != Phase::Input && phase != Phase::Output) ||
      (phase == Phase::Input && (input_issued_ || native_done_)) ||
      (phase == Phase::Output && (!native_done_ || output_issued_))) {
    return false;
  }
  Node node{};
  if (!plan_->project(NodeId{.epoch = 0u, .phase = phase}, node) ||
      !authority_->executions().issue_execution(lease_.token, lease_.generation, *plan_,
                                   node, sequence_, ticket)) {
    return false;
  }
  ++sequence_;
  if (phase == Phase::Input) {
    input_issued_ = true;
    input_ = ticket;
  } else {
    output_issued_ = true;
    output_ = ticket;
  }
  return true;
}

bool Run::terminal(const ExecutionTicket &ticket,
                   const ExecutionTerminal terminal,
                   const Status status) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ ||
      (terminal == ExecutionTerminal::Success) != static_cast<bool>(status)) {
    return false;
  }
  const bool input = input_issued_ && !input_done_ &&
                     ticket.sequence == input_.sequence &&
                     ticket.phase == static_cast<std::uint8_t>(Phase::Input);
  const bool output = output_issued_ && !output_done_ &&
                      ticket.sequence == output_.sequence &&
                      ticket.phase == static_cast<std::uint8_t>(Phase::Output);
  if ((!input && !output) ||
      !authority_->executions().terminal_execution(ticket, terminal)) {
    return false;
  }
  if (input) {
    input_done_ = true;
  } else {
    output_done_ = true;
  }
  if (terminal == ExecutionTerminal::Success) {
    return true;
  }
  host_status_ = status;
  host_unknown_ =
      host_unknown_ || terminal == ExecutionTerminal::UnknownMayWrite;
  const std::uint8_t phase = ticket.phase;
  const std::uint8_t mask = static_cast<std::uint8_t>(1u << phase);
  if (host_failure_count_ == host_failures_.size()) {
    host_unknown_ = true;
    return true;
  }
  host_failures_[host_failure_count_++] = FailureEvidence{
      .epoch = ticket.epoch,
      .phases = mask,
      .may_write = ticket.may_write ? mask : std::uint8_t{0u},
      .terminal = mask,
  };
  return true;
}

bool Run::native(const NativeEvidence &evidence) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || native_done_ ||
      !input_done_ || !authority_->executions().accept_execution(*plan_, evidence)) {
    return false;
  }
  native_ = evidence;
  native_done_ = true;
  // Authority consumes one opaque native terminal as the second issue in Q=1.
  ++sequence_;
  return true;
}

bool Run::reject(const Status failure) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || native_done_ ||
      native_rejected_ || !input_done_ || failure ||
      !authority_->executions().reject_execution(lease_.token, lease_.generation, *plan_,
                                    failure)) {
    return false;
  }
  native_rejected_ = true;
  host_status_ = failure;
  return true;
}

bool Run::abandon() noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ ||
      !authority_->executions().abandon_execution(lease_.token, lease_.generation, *plan_)) {
    return false;
  }
  closed_ = true;
  return true;
}

} // namespace rund::compute::detail::residency::execution
