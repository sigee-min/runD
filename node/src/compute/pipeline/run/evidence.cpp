#include "evidence.hpp"

#include "../../status.hpp"

#include <algorithm>

namespace rund::compute::detail {
namespace {
[[nodiscard]] PipelineNativeCompletion
native_completion(const node::accel::detail::PreparedPipelineEvidence &evidence,
                  const bool submitted) noexcept {
  if (evidence.terminal != node::accel::detail::NativeTerminal::Known) {
    return PipelineNativeCompletion::UnknownMayWrite;
  }
  return submitted ? PipelineNativeCompletion::Known
                   : PipelineNativeCompletion::NotSubmitted;
}
[[nodiscard]] bool valid_control_identity(
    const PipelineEvidenceContext &context,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  return evidence.control_observed && evidence.control_valid &&
         evidence.control_byte_count ==
             node::accel::detail::PreparedPipelineControlBytes &&
         node::accel::detail::PreparedPipelineGenerationMatches(
             evidence.control, context.generation + 1u);
}
} // namespace

PipelineEvidenceDecision decide_ordinary_pipeline_evidence(
    const PipelineEvidenceContext &context,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  PipelineEvidenceDecision decision{};
  PipelineOutcome &result = decision.outcome;
  result.status = evidence.check.ok
                      ? Status::success()
                      : Status::fail(project_reason(evidence.check.reason,
                                                    Reason::BackendFailed));
  decision.layout_valid = context.active_steps == evidence.active_step_count;
  if (!decision.layout_valid) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }
  const bool submitted = evidence.submitted || context.prior_submission ||
                         evidence.shared.run.work.command_submit_count != 0u;
  result.native_completion = native_completion(evidence, submitted);
  const std::uint64_t expected_submits = context.timed_metal ? 2u : 1u;
  const bool submit_count_valid =
      result.status
          ? evidence.shared.run.work.command_submit_count == expected_submits
          : evidence.shared.run.work.command_submit_count != 0u &&
                evidence.shared.run.work.command_submit_count <=
                    expected_submits;
  if (result.submitted() && !submit_count_valid) {
    result.status = Status::fail(Reason::CompletionInvalid);
  } else if (!result.submitted() &&
             evidence.shared.run.work.command_submit_count != 0u) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }

  decision.control_valid = valid_control_identity(context, evidence);
  if (evidence.control_observed && !decision.control_valid) {
    result.status = Status::fail(Reason::CompletionInvalid);
  } else if (decision.control_valid) {
    result.verified = std::min<std::size_t>(evidence.control.verified_prefix,
                                            context.total_steps);
    result.failed_step.reset();
    if (!result.status &&
        evidence.control.failed_step !=
            node::accel::detail::PreparedPipelineNoStep &&
        evidence.control.failed_step < context.total_steps) {
      result.failed_step = evidence.control.failed_step;
    }
    result.publication_suppressed = evidence.control.reason != 0u;
  } else if (result.status && !result.submitted()) {
    result.verified = context.total_steps;
  }
  if (decision.control_valid &&
      !node::accel::detail::DecodePipelineNestedPhase(
          evidence.control.failed_nested_phase, decision.failed_nested_phase)) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }
  if (result.unknown_terminal()) {
    result.status = Status::fail(Reason::DeviceLost);
    result.publication_suppressed = true;
  }
  result.writes_possible = result.submitted();
  return decision;
}

PipelineEvidenceDecision decide_residency_pipeline_evidence(
    const PipelineEvidenceContext &context,
    const node::accel::detail::PreparedPipelineEvidence &evidence,
    PipelineOutcome prepared) noexcept {
  PipelineEvidenceDecision decision{.outcome = prepared};
  PipelineOutcome &outcome = decision.outcome;
  outcome.status =
      evidence.check.ok && evidence.shared.outcome.ok
          ? Status::success()
          : Status::fail(project_reason(!evidence.check.ok
                                            ? evidence.check.reason
                                            : evidence.shared.outcome.reason,
                                        Reason::BackendFailed));
  const bool submitted =
      evidence.submitted || evidence.shared.run.work.command_submit_count != 0u;
  outcome.native_completion = native_completion(evidence, submitted);
  const bool known =
      evidence.terminal == node::accel::detail::NativeTerminal::Known;
  decision.layout_valid = context.active_steps != 0u &&
                          evidence.active_step_count == context.active_steps;
  const bool submit_valid =
      evidence.shared.run.work.command_submit_count <= 1u &&
      (!outcome.status ||
       (outcome.submitted() &&
        evidence.shared.run.work.command_submit_count == 1u && known));
  decision.control_valid =
      valid_control_identity(context, evidence) &&
      evidence.control.verified_prefix <= context.active_steps;
  if (!decision.layout_valid || !submit_valid ||
      (evidence.control_observed && !decision.control_valid) ||
      (outcome.status &&
       (!decision.control_valid ||
        evidence.control.verified_prefix != context.active_steps ||
        evidence.control.failed_step !=
            node::accel::detail::PreparedPipelineNoStep ||
        evidence.control.reason != static_cast<std::uint32_t>(Reason::Ok)))) {
    outcome.status = Status::fail(Reason::CompletionInvalid);
  }

  if (!known) {
    // An accepted native command without a Known terminal cannot prove that
    // its control row or output was not written. Keep the Pipeline and its
    // Registry lease quarantined instead of allowing a reusable identity.
    outcome.status = Status::fail(Reason::DeviceLost);
    outcome.publication_suppressed = true;
  }

  if (decision.control_valid) {
    outcome.verified = evidence.control.verified_prefix;
    outcome.failed_step.reset();
    if (!outcome.status &&
        evidence.control.failed_step !=
            node::accel::detail::PreparedPipelineNoStep &&
        evidence.control.failed_step < context.total_steps) {
      outcome.failed_step = evidence.control.failed_step;
    }
    outcome.publication_suppressed =
        outcome.publication_suppressed || evidence.control.reason != 0u;
  } else {
    outcome.publication_suppressed = true;
  }
  outcome.writes_possible = outcome.writes_possible || outcome.submitted();
  return decision;
}

} // namespace rund::compute::detail
