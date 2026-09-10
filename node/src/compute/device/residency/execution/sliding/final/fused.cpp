#include "abort.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] ExecutionClose
close_from_abort(const FinalAbort result) noexcept {
  switch (result) {
  case FinalAbort::Closed:
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = false};
  case FinalAbort::Quarantined:
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = true};
  case FinalAbort::Invalid:
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  return ExecutionClose{.failure = AuthorityFailure::Invalid};
}

} // namespace

ExecutionClose SlidingOwner::commit_sliding_final(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingFinal &final, ExecutionSlidingFinal &&prepared,
    execution::SlidingEvidence &evidence, void *const publication,
    const ExecutionSlidingPublication commit) noexcept {
  evidence = {};
  const bool callback = publication != nullptr || commit != nullptr;
  if (sliding.state_ == nullptr || !final || !prepared ||
      (callback && (publication == nullptr || commit == nullptr)) ||
      (callback && !prepared.success_)) {
    if (sliding.state_ != nullptr && final && prepared) {
      const FinalAbort aborted =
          abort_execution_sliding(plan, sliding, final, &prepared,
                                  Status::fail(Reason::CompletionInvalid));
      if (aborted != FinalAbort::Invalid) {
        return close_from_abort(aborted);
      }
    }
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }

  // Authority -> Sliding is the sole paired lock order. Every credential and
  // Final-state check precedes frame retirement, state close, or slot reset.
  std::unique_lock authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = true};
  }
  // A fused continuation owns the live credential until its callback returns.
  // A duplicate finish, including a raw close, is a no-mutation Busy result.
  if (authority_.execution_state_.slot.sliding_final_phase !=
      SlidingFinalPhase::None) {
    return ExecutionClose{.failure = AuthorityFailure::Busy};
  }
  execution::Sliding::State &state = *sliding.state_;
  std::unique_lock sliding_lock{state.gate};
  if (!sliding_final_matches(plan, sliding, final, prepared,
                             SlidingFinalPhase::None)) {
    sliding_lock.unlock();
    authority_lock.unlock();
    const FinalAbort aborted =
        abort_execution_sliding(plan, sliding, final, &prepared,
                                Status::fail(Reason::CompletionInvalid));
    if (aborted != FinalAbort::Invalid) {
      return close_from_abort(aborted);
    }
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }

  if (!commit_sliding_frames(&plan, prepared.success_)) {
    sliding_lock.unlock();
    authority_lock.unlock();
    const FinalAbort aborted =
        abort_execution_sliding(plan, sliding, final, &prepared,
                                Status::fail(Reason::CompletionInvalid));
    if (aborted != FinalAbort::Invalid) {
      return close_from_abort(aborted);
    }
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  const bool success = prepared.success_;
  evidence = final.evidence_;
  if (!callback) {
    if (state.has_failure) {
      state.inputs = {};
      state.outputs = {};
      state.native_cells = {};
      state.terminals = {};
    }
    state.closed = true;
    state.finalizing = false;
    state.final_nonce = 0u;
    state.authority_bound = false;
    prepared = {};
    authority_.execution_state_.slot = registry_model::ExecutionSlot{};
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = success,
                          .quarantined = false};
  }

  // Keep Pipeline/publication/backing gates owned by the caller, but release
  // Authority and Sliding while the private noexcept continuation runs. The
  // token, Final nonce, rows, and owner remain live so Authority reentry is
  // immediately Busy rather than a deadlock or a second disposition.
  authority_.execution_state_.slot.sliding_final_phase =
      SlidingFinalPhase::Inflight;
  authority_lock.unlock();
  sliding_lock.unlock();
  commit(publication);

  // Reacquire in Authority -> Sliding order and authenticate the exact live
  // continuation. Publication has already happened, so any contradiction is
  // sticky quarantine with no callback retry or rollback.
  authority_lock.lock();
  sliding_lock.lock();
  if (authority_.view_commit_quarantined_locked()) {
    quarantine_sliding_locked(sliding, Status::fail(Reason::CompletionInvalid));
    prepared = {};
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = true};
  }
  if (!sliding_final_matches(plan, sliding, final, prepared,
                             SlidingFinalPhase::Inflight)) {
    quarantine_sliding_locked(sliding, Status::fail(Reason::CompletionInvalid));
    prepared = {};
    return ExecutionClose{.failure = AuthorityFailure::None,
                          .success = false,
                          .quarantined = true};
  }

  state.closed = true;
  state.finalizing = false;
  state.final_nonce = 0u;
  state.authority_bound = false;
  prepared = {};
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return ExecutionClose{.failure = AuthorityFailure::None,
                        .success = success,
                        .quarantined = false};
}

} // namespace rund::compute::detail::residency
