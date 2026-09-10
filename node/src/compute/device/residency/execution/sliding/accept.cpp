#include "final/internal.hpp"

namespace rund::compute::detail::residency {
ExecutionClose SlidingOwner::accept_execution_sliding_final(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingFinal &final, ExecutionSlidingFinal &&prepared,
    execution::SlidingEvidence &evidence, void *const publication,
    const ExecutionSlidingPublication commit) noexcept {
  return commit_sliding_final(plan, sliding, final, std::move(prepared),
                              evidence, publication, commit);
}

} // namespace rund::compute::detail::residency
