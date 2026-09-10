#include "../../run/evidence.hpp"
#include "../submit.hpp"

#include "../../claim.hpp"
#include "../../local.hpp"
#include "../../state.hpp"

#include "../../../stats.hpp"
#include "../../../status.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail {

PipelineOutcome finish_residency_pipeline_execution(
    PipelineState &state, const std::span<const std::uint32_t> locals,
    const node::accel::detail::PreparedPipelineEvidence &evidence,
    PipelineOutcome outcome) noexcept {
  const PipelineEvidenceDecision decision = decide_residency_pipeline_evidence(
      {.total_steps = state.steps.size(),
       .active_steps = locals.size(),
       .generation = state.attempt.generation},
      evidence, outcome);
  accumulate_run_facts(state.stats, evidence.shared.run);
  state.stats.pipeline.status_entry_count = evidence.status_entry_count;
  state.stats.pipeline.control_byte_count = evidence.control_byte_count;
  state.stats.pipeline.control_command_count = evidence.control_command_count;
  state.stats.pipeline.control_ns = evidence.control_ns;

  outcome = decision.outcome;
  state.attempt.backend_submitted = outcome.submitted();
  if (outcome.unknown_terminal()) {
    state.control_poisoned = true;
  }
  if (!outcome.status && outcome.failed_step.has_value()) {
    outcome.status = pipeline_window_status(
        state.windows, state.steps[outcome.failed_step.value_or(0u)],
        outcome.status, state.stats.control);
  }
  capture_accel_pipeline_profile(state, evidence, false);
  if (!outcome.status &&
      !rebase_failed_pipeline_generation(state, outcome.submitted(),
                                         outcome.status.reason())) {
    outcome.status = Status::fail(Reason::PipelinePoisoned);
  }
  state.stats.pipeline.verified_step_count = outcome.verified;
  state.stats.pipeline.failed_step_index =
      outcome.failed_step.has_value()
          ? logical_step_index(state, outcome.failed_step.value_or(0u))
          : PipelineStats::no_failed_step;
  state.attempt.verified = outcome.verified;
  state.attempt.failure_step_known = outcome.failed_step.has_value();
  state.attempt.writes_possible =
      outcome.writes_possible || outcome.submitted();
  outcome.writes_possible = state.attempt.writes_possible;
  return outcome;
}

Status
publish_residency_pipeline_execution(PipelineState &state,
                                     const std::size_t issued_steps,
                                     const PipelineOutcome &outcome) noexcept {
  constexpr PipelineClaimAuthority authority =
      PipelineClaimAuthority::PrivateResidency;
  if (outcome.status) {
    publish_pipeline_terminal(
        state, PipelineTerminal{.issued_steps = issued_steps}, authority);
  } else {
    publish_pipeline_terminal(
        state,
        PipelineTerminal{
            .reason = outcome.status.reason(),
            .verified = outcome.verified,
            .issued_steps = issued_steps,
            .failed_step = outcome.failed_step.value_or(0u),
            .failure_step_known = outcome.failed_step.has_value(),
            .writes_possible = outcome.writes_possible || outcome.submitted(),
            .publication_suppressed = outcome.publication_suppressed,
        },
        authority);
  }
  return state.failure == Reason::Ok && state.phase == PipelinePhase::Ready
             ? Status::success()
             : Status::fail(state.failure == Reason::Ok
                                ? Reason::PipelinePoisoned
                                : state.failure);
}

} // namespace rund::compute::detail
