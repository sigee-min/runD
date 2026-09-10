#include "../../registry.hpp"
#include "../execution_owner.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"

#include "../../execution/plan.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::residency {

ExecutionLease
ExecutionOwner::begin_execution(const execution::Plan &plan) noexcept {
  return authority_.admit_execution(plan, false, false);
}

ExecutionLease
ExecutionOwner::begin_execution_window(const execution::Plan &plan) noexcept {
  return authority_.admit_execution(plan, true, false);
}

ExecutionLease Authority::admit_execution(const execution::Plan &plan,
                                          const bool window,
                                          const bool sliding) noexcept {
  std::lock_guard lock{gate_};
  const bool busy = execution_state_.slot.token != 0u ||
                    cycle_state_.cycle.token != 0u ||
                    active(cycle_state_.writeback) ||
                    any_active(cycle_state_.graph_persists) ||
                    std::any_of(cycle_state_.epochs.begin(),
                                cycle_state_.epochs.end(), active);
  if (view_commit_quarantined_locked() || plan.identity() == 0u ||
      plan.epoch_count() == 0u ||
      plan.epoch_count() > std::numeric_limits<std::uint64_t>::max() / 3u ||
      busy) {
    return ExecutionLease{.failure = busy ? AuthorityFailure::Busy
                                          : AuthorityFailure::Invalid};
  }

  // Reserve both credentials before candidate journal or frame mutation. An
  // exhausted credential leaves admission byte-for-byte idle.
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (credentials_.next_token == 0u || credentials_.next_token > max - 2u) {
    return ExecutionLease{.failure = AuthorityFailure::Capacity};
  }
  const std::uint64_t token = next_token(credentials_.next_token);
  const std::uint64_t generation = next_token(credentials_.next_token);
  if (token == 0u || generation == 0u) {
    return ExecutionLease{.failure = AuthorityFailure::Capacity};
  }

  registry_model::ExecutionSlot candidate{};
  candidate.token = token;
  candidate.generation = generation;
  for (auto &bank : candidate.issued) {
    bank.fill(NeverUse);
  }
  for (auto &bank : candidate.terminals) {
    bank.fill(NeverUse);
  }
  candidate.plan = plan.identity();
  candidate.epochs = plan.epoch_count();
  candidate.sliding_host_input_regions = plan.host_input_regions();

  if (!validate_execution_regions_locked(plan, sliding, candidate)) {
    return ExecutionLease{.failure = AuthorityFailure::Invalid};
  }
  if (plan.epoch_count() == 1u && !sliding) {
    const AuthorityFailure failure =
        admit_direct_execution_locked(plan, candidate);
    if (failure != AuthorityFailure::None) {
      return ExecutionLease{.failure = failure};
    }
  } else if (window) {
    // Window rows are admitted at each Host-service issue; mutually exclusive
    // epochs are never pinned in the same two physical banks at begin.
    candidate.window_cache_admitted = true;
  }
  candidate.sliding_admitted = sliding;
  execution_state_.slot = candidate;
  return ExecutionLease{.failure = AuthorityFailure::None,
                        .token = execution_state_.slot.token,
                        .generation = execution_state_.slot.generation,
                        .owner_nonce = execution_state_.slot.owner_nonce,
                        .plan = execution_state_.slot.plan,
                        .epochs = execution_state_.slot.epochs};
}

} // namespace rund::compute::detail::residency
