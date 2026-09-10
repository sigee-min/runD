#include "../internal.hpp"

#include "../../claim.hpp"
#include "../../state.hpp"

namespace rund::compute::detail {

Status publish_residency_pipeline_outcome(
    PipelineState &state, const std::size_t issued_steps,
    const PipelineOutcome &outcome, const bool defer_generation,
    const std::uint64_t publication_generation,
    const std::uint8_t publication_parity) noexcept {
  const bool deferred = defer_generation || outcome.defer_generation;
  PipelineTerminal terminal{.issued_steps = issued_steps,
                            .defer_generation = deferred,
                            .publication_generation = publication_generation,
                            .publication_parity = publication_parity};
  if (!outcome.status) {
    terminal.reason = outcome.status.reason();
    terminal.verified = outcome.verified;
    terminal.failed_step = outcome.failed_step.value_or(0u);
    terminal.failure_step_known = outcome.failed_step.has_value();
    terminal.writes_possible = outcome.writes_possible || outcome.submitted();
    terminal.publication_suppressed = outcome.publication_suppressed;
  }
  publish_pipeline_terminal(state, terminal,
                            PipelineClaimAuthority::PrivateResidency);
  return state.failure == Reason::Ok && state.phase == PipelinePhase::Ready
             ? Status::success()
             : Status::fail(state.failure == Reason::Ok
                                ? Reason::PipelinePoisoned
                                : state.failure);
}

} // namespace rund::compute::detail
