#include "authority_close/internal.hpp"

#include "../../registry/execution_owner.hpp"

namespace rund::compute::detail::residency {

ExecutionClose
ExecutionOwner::close_execution(const execution::Plan &plan,
                                const execution::Evidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  const execution::authority_close_detail::Validation validation =
      execution::authority_close_detail::validate(
          authority_.execution_state_.slot, plan, evidence,
          !authority_.view_commit_quarantined_locked() &&
              authority_.close_rows_clear_locked());
  if (validation.failure != AuthorityFailure::None) {
    return ExecutionClose{.failure = validation.failure};
  }
  const execution::authority_close_detail::Preparation prepared =
      execution::authority_close_detail::prepare(
          authority_.frames_, authority_.execution_state_.slot, plan, evidence,
          validation.successful);
  if (prepared.failure != AuthorityFailure::None) {
    return ExecutionClose{.failure = prepared.failure};
  }
  return execution::authority_close_detail::apply(
      authority_.frames_, authority_.execution_state_.slot, evidence,
      validation, prepared);
}

} // namespace rund::compute::detail::residency
