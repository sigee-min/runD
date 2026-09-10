#include "sliding_owner.hpp"

#include "../execution/plan.hpp"

#include <mutex>

namespace rund::compute::detail::residency {

SlidingOwner::SlidingOwner(Authority &authority) noexcept
    : authority_(authority) {}

SlidingOwner Authority::sliding() noexcept { return SlidingOwner{*this}; }

ExecutionLease
SlidingOwner::begin_execution_sliding(const execution::Plan &plan) noexcept {
  return authority_.admit_execution(plan, true, true);
}

bool SlidingOwner::abandon_execution_sliding(
    const execution::Plan &plan, const ExecutionLease &lease) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      authority_.execution_state_.slot.token == 0u ||
      lease.token != authority_.execution_state_.slot.token ||
      lease.generation != authority_.execution_state_.slot.generation ||
      (lease.owner_nonce != 0u &&
       lease.owner_nonce != authority_.execution_state_.slot.owner_nonce) ||
      lease.plan != authority_.execution_state_.slot.plan ||
      lease.epochs != authority_.execution_state_.slot.epochs ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      plan.epoch_count() != authority_.execution_state_.slot.epochs ||
      authority_.execution_state_.slot.next_sequence != 1u ||
      authority_.execution_state_.slot.release_count != 0u ||
      authority_.execution_state_.slot.native_inflight != 0u) {
    return false;
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

bool SlidingOwner::seal_execution_sliding_lease(
    ExecutionLease &lease) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.native_inflight == 0u ||
      lease.token != authority_.execution_state_.slot.token ||
      lease.generation != authority_.execution_state_.slot.generation ||
      lease.plan != authority_.execution_state_.slot.plan ||
      lease.epochs != authority_.execution_state_.slot.epochs ||
      authority_.execution_state_.slot.owner_nonce == 0u) {
    return false;
  }
  lease.owner_nonce = authority_.execution_state_.slot.owner_nonce;
  return true;
}

bool SlidingOwner::validate_execution_sliding_lease(
    const execution::Plan &plan, const ExecutionLease &lease) noexcept {
  std::lock_guard lock{authority_.gate_};
  return !authority_.view_commit_quarantined_locked() &&
         authority_.execution_state_.slot.sliding_admitted &&
         !authority_.execution_state_.slot.window_final &&
         authority_.execution_state_.slot.token != 0u &&
         authority_.execution_state_.slot.native_inflight != 0u &&
         authority_.execution_state_.slot.owner_nonce != 0u &&
         lease.token == authority_.execution_state_.slot.token &&
         lease.generation == authority_.execution_state_.slot.generation &&
         lease.owner_nonce == authority_.execution_state_.slot.owner_nonce &&
         lease.plan == authority_.execution_state_.slot.plan &&
         lease.epochs == authority_.execution_state_.slot.epochs &&
         plan.identity() == authority_.execution_state_.slot.plan &&
         plan.epoch_count() == authority_.execution_state_.slot.epochs;
}

} // namespace rund::compute::detail::residency
