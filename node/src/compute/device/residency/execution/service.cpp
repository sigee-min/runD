#include "service.hpp"

#include "../registry/execution_owner.hpp"
#include "window.hpp"

namespace rund::compute::detail::residency::execution {

bool Service::bind(Authority &authority, const Plan &plan,
                   const ExecutionLease &lease) noexcept {
  if (authority_ != nullptr || !lease || plan.identity() == 0u ||
      plan.epoch_count() == 0u ||
      lease.plan != plan.identity() || lease.epochs != plan.epoch_count()) {
    return false;
  }
  authority_ = &authority;
  plan_ = &plan;
  lease_ = lease;
  return true;
}

bool Service::issue(const std::uint64_t epoch, const Phase phase,
                    ExecutionTicket &ticket) noexcept {
  ticket = {};
  if (authority_ == nullptr || plan_ == nullptr || epoch >= lease_.epochs ||
      (phase != Phase::Input && phase != Phase::Output)) {
    return false;
  }
  Node node{};
  if (!plan_->project(NodeId{.epoch = epoch, .phase = phase}, node) ||
      !authority_->executions().issue_execution(lease_.token, lease_.generation, *plan_,
                                   node, sequence_, ticket)) {
    return false;
  }
  ++sequence_;
  return true;
}

bool Service::terminal(const ExecutionTicket &ticket,
                       const ExecutionTerminal terminal,
                       const Status status) noexcept {
  return authority_ != nullptr && plan_ != nullptr &&
         (terminal == ExecutionTerminal::Success) ==
             static_cast<bool>(status) &&
         authority_->executions().terminal_execution(ticket, terminal);
}

bool Service::release(const Release &release) noexcept {
  if (authority_ == nullptr || plan_ == nullptr ||
      !authority_->executions().release_execution(lease_.token, lease_.generation, *plan_,
                                     release, sequence_)) {
    return false;
  }
  ++sequence_;
  return true;
}

} // namespace rund::compute::detail::residency::execution
