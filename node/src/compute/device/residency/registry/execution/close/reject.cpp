#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"

#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::reject_execution(const std::uint64_t token,
                                      const std::uint64_t generation,
                                      const execution::Plan &plan,
                                      const Status failure) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  constexpr std::uint8_t DispatchMask =
      std::uint8_t{1u} << static_cast<std::uint8_t>(execution::Phase::Dispatch);
  if (failure || token == 0u || generation == 0u ||
      authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.generation != generation ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != 1u ||
      plan.epoch_count() != 1u ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.failed ||
      authority_.execution_state_.slot.progress.input_services != 1u ||
      authority_.execution_state_.slot.progress.native_dispatches != 0u ||
      authority_.execution_state_.slot.progress.native_completions != 0u ||
      authority_.execution_state_.slot.progress.output_services != 0u ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      authority_.execution_state_.slot.next_sequence != 2u ||
      authority_.execution_state_.slot.failure_count != 0u) {
    return false;
  }
  authority_.execution_state_.slot.failures[0u] = ExecutionFailure{
      .epoch = 0u,
      .phases = DispatchMask,
      .may_write = 0u,
      .terminal = DispatchMask,
  };
  authority_.execution_state_.slot.failure_count = 1u;
  authority_.execution_state_.slot.failed = true;
  authority_.execution_state_.slot.native_rejected = true;
  return true;
}

} // namespace rund::compute::detail::residency
