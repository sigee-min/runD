#include "internal.hpp"

namespace rund::compute::detail::residency {

ExecutionClose SlidingOwner::accept_execution_sliding(
    ExecutionSlidingFinal &&prepared,
    ExecutionSlidingReceipt &receipt) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = true};
  }
  if (authority_.execution_state_.slot.sliding_final_phase !=
      SlidingFinalPhase::None) {
    return ExecutionClose{.failure = AuthorityFailure::Busy};
  }
  if (receipt || !prepared ||
      !authority_.execution_state_.slot.sliding_admitted ||
      !authority_.execution_state_.slot.window_final ||
      prepared.plan_ != authority_.execution_state_.slot.plan ||
      prepared.token_ != authority_.execution_state_.slot.token ||
      prepared.generation_ != authority_.execution_state_.slot.generation ||
      prepared.owner_ != authority_.execution_state_.slot.native_inflight ||
      prepared.sliding_nonce_ == 0u ||
      prepared.nonce_ != authority_.execution_state_.slot.next_sequence) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }

  if (!commit_sliding_frames(nullptr, prepared.success_)) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }

  receipt.plan_ = prepared.plan_;
  receipt.token_ = prepared.token_;
  receipt.generation_ = prepared.generation_;
  receipt.owner_ = prepared.owner_;
  receipt.sliding_nonce_ = prepared.sliding_nonce_;
  receipt.nonce_ = prepared.nonce_;
  receipt.success_ = prepared.success_;
  prepared = {};
  const bool success = receipt.success_;
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return ExecutionClose{.failure = AuthorityFailure::None,
                        .success = success,
                        .quarantined = false};
}

ExecutionClose SlidingOwner::accept_execution_sliding(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingFinal &final, ExecutionSlidingFinal &&prepared,
    execution::SlidingEvidence &evidence) noexcept {
  return commit_sliding_final(plan, sliding, final, std::move(prepared),
                              evidence, nullptr, nullptr);
}

} // namespace rund::compute::detail::residency
