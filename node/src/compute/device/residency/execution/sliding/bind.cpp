#include "internal.hpp"

#include <atomic>
#include <limits>

namespace rund::compute::detail::residency {

namespace {

std::atomic<std::uint64_t> NextSlidingAuthorityNonce{1u};

[[nodiscard]] bool mint_nonce(std::uint64_t &nonce) noexcept {
  std::uint64_t current =
      NextSlidingAuthorityNonce.load(std::memory_order_relaxed);
  for (;;) {
    if (current == 0u || current == std::numeric_limits<std::uint64_t>::max()) {
      nonce = 0u;
      return false;
    }
    if (NextSlidingAuthorityNonce.compare_exchange_weak(
            current, current + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      nonce = current;
      return true;
    }
  }
}

} // namespace

bool SlidingOwner::bind_execution_sliding(
    execution::Sliding &sliding) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      authority_.execution_state_.slot.owner_nonce != 0u ||
      authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.sliding_final_phase !=
          SlidingFinalPhase::None) {
    return false;
  }
  std::uint64_t nonce = 0u;
  if (!mint_nonce(nonce)) {
    authority_.execution_state_.slot = registry_model::ExecutionSlot{};
    return false;
  }
  std::uint64_t owner = 0u;
  if (!sliding.bind_authority(
          residency::Identity{.lo = authority_.execution_state_.slot.plan},
          authority_.execution_state_.slot.token,
          authority_.execution_state_.slot.generation, owner) ||
      owner == 0u || owner == std::numeric_limits<std::uint64_t>::max()) {
    authority_.execution_state_.slot = registry_model::ExecutionSlot{};
    return false;
  }
  authority_.execution_state_.slot.native_inflight = owner;
  authority_.execution_state_.slot.owner_nonce = nonce;
  return true;
}

bool SlidingOwner::abandon_bound_execution_sliding(
    const execution::Plan &plan, execution::Sliding &sliding,
    const ExecutionLease &lease) noexcept {
  if (sliding.state_ == nullptr) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard sliding_lock{state.gate};
  const bool identity =
      authority_.execution_state_.slot.sliding_admitted &&
      !authority_.execution_state_.slot.window_final &&
      authority_.execution_state_.slot.token != 0u &&
      authority_.execution_state_.slot.generation != 0u &&
      authority_.execution_state_.slot.native_inflight != 0u &&
      authority_.execution_state_.slot.owner_nonce != 0u &&
      state.authority_bound && !state.closed &&
      state.owner == authority_.execution_state_.slot.native_inflight &&
      state.plan.lo == authority_.execution_state_.slot.plan &&
      state.token == authority_.execution_state_.slot.token &&
      state.generation == authority_.execution_state_.slot.generation &&
      plan.identity() == authority_.execution_state_.slot.plan &&
      plan.epoch_count() == authority_.execution_state_.slot.epochs;
  const auto matches = [&]() noexcept {
    return identity && !state.quarantine &&
           lease.token == authority_.execution_state_.slot.token &&
           lease.generation == authority_.execution_state_.slot.generation &&
           lease.owner_nonce == authority_.execution_state_.slot.owner_nonce &&
           lease.plan == authority_.execution_state_.slot.plan &&
           lease.epochs == authority_.execution_state_.slot.epochs;
  };
  const auto quarantine = [&](const Status failure) noexcept {
    state.quarantine_bound(failure);
    authority_.execution_state_.slot.failed = true;
    authority_.execution_state_.slot.unknown = true;
    authority_.execution_state_.slot.sliding_final_phase =
        SlidingFinalPhase::Quarantined;
  };
  if (authority_.execution_state_.slot.sliding_final_phase !=
          SlidingFinalPhase::None ||
      state.finalizing) {
    return false;
  }
  if (!matches()) {
    return false;
  }
  if (authority_.execution_state_.slot.next_sequence != 1u ||
      authority_.execution_state_.slot.release_count != 0u ||
      !state.no_issued() || !state.callbacks_quiesced() ||
      !state.abandon_bound()) {
    quarantine(Status::fail(Reason::CompletionInvalid));
    return false;
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

bool SlidingOwner::quarantine_bound_execution_sliding(
    const execution::Plan &plan, execution::Sliding &sliding,
    const Status failure) noexcept {
  if (sliding.state_ == nullptr) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard sliding_lock{state.gate};
  const bool identity =
      authority_.execution_state_.slot.sliding_admitted &&
      !authority_.execution_state_.slot.window_final &&
      authority_.execution_state_.slot.token != 0u &&
      authority_.execution_state_.slot.generation != 0u &&
      authority_.execution_state_.slot.native_inflight != 0u &&
      authority_.execution_state_.slot.owner_nonce != 0u &&
      state.authority_bound && !state.closed &&
      state.owner == authority_.execution_state_.slot.native_inflight &&
      state.plan.lo == authority_.execution_state_.slot.plan &&
      state.token == authority_.execution_state_.slot.token &&
      state.generation == authority_.execution_state_.slot.generation &&
      plan.identity() == authority_.execution_state_.slot.plan &&
      plan.epoch_count() == authority_.execution_state_.slot.epochs;
  if (!identity) {
    return false;
  }
  if (authority_.execution_state_.slot.sliding_final_phase !=
          SlidingFinalPhase::None &&
      authority_.execution_state_.slot.sliding_final_phase !=
          SlidingFinalPhase::Quarantined) {
    return false;
  }
  if (!state.quarantine) {
    state.quarantine_bound(failure);
  }
  authority_.execution_state_.slot.failed = true;
  authority_.execution_state_.slot.unknown = true;
  authority_.execution_state_.slot.sliding_final_phase =
      SlidingFinalPhase::Quarantined;
  return true;
}

} // namespace rund::compute::detail::residency
