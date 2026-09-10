#include "internal.hpp"

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] std::uint64_t next_token(std::uint64_t &next) noexcept {
  const std::uint64_t token = next++;
  if (next == 0u) {
    next = 1u;
  }
  return token;
}

} // namespace

bool SlidingOwner::prepare_execution_sliding(
    const execution::Plan &plan, const execution::SlidingFinal &final,
    ExecutionSlidingFinal &prepared) noexcept {
  prepared = {};
  if (!final) {
    return false;
  }
  const execution::SlidingEvidence &evidence = final.evidence();
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const bool untouched =
      authority_.execution_state_.slot.next_sequence == 1u &&
      authority_.execution_state_.slot.progress.input_services == 0u &&
      authority_.execution_state_.slot.progress.native_dispatches == 0u &&
      authority_.execution_state_.slot.progress.native_completions == 0u &&
      authority_.execution_state_.slot.progress.output_services == 0u &&
      !authority_.execution_state_.slot.native_accepted &&
      !authority_.execution_state_.slot.native_rejected &&
      authority_.execution_state_.slot.release_count == 0u &&
      authority_.execution_state_.slot.window_accept_count == 0u;
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final || !untouched ||
      authority_.execution_state_.slot.token == 0u ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      plan.epoch_count() != authority_.execution_state_.slot.epochs ||
      evidence.plan.hi != 0u ||
      evidence.plan.lo != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      authority_.execution_state_.slot.native_inflight == 0u ||
      evidence.owner != authority_.execution_state_.slot.native_inflight ||
      evidence.planned != authority_.execution_state_.slot.epochs ||
      evidence.first_unsent != evidence.admitted ||
      evidence.admitted > evidence.planned ||
      evidence.terminal_frontier > evidence.admitted ||
      evidence.persist_frontier > evidence.persist_issued ||
      evidence.quarantined ||
      evidence.terminal == execution::TerminalKind::UnknownMayWrite) {
    return false;
  }
  const bool success = static_cast<bool>(evidence.status);
  if (success) {
    if (evidence.has_failure ||
        evidence.first_failure_terminal != execution::TerminalKind::Known ||
        evidence.terminal != execution::TerminalKind::Known ||
        evidence.admitted != evidence.planned ||
        evidence.terminal_frontier != evidence.admitted ||
        evidence.persist_frontier != evidence.persist_issued) {
      return false;
    }
  } else if (!evidence.has_failure ||
             evidence.first_failure.ordinal >= evidence.planned ||
             evidence.first_failure_terminal !=
                 execution::TerminalKind::Known ||
             evidence.terminal != execution::TerminalKind::Known) {
    return false;
  }

  const std::uint64_t nonce = next_token(authority_.credentials_.next_token);
  authority_.execution_state_.slot.next_sequence = nonce;
  authority_.execution_state_.slot.window_final = true;
  prepared.plan_ = authority_.execution_state_.slot.plan;
  prepared.token_ = authority_.execution_state_.slot.token;
  prepared.generation_ = authority_.execution_state_.slot.generation;
  prepared.owner_ = evidence.owner;
  prepared.sliding_nonce_ = final.nonce_;
  prepared.nonce_ = nonce;
  prepared.success_ = success;
  return true;
}

} // namespace rund::compute::detail::residency
