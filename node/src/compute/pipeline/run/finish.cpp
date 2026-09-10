#include "evidence.hpp"
#include "result.hpp"

#include "../local.hpp"

#include "../../stats.hpp"
#include "../../status.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail {
void reset_pipeline_stats(PipelineState &state) noexcept {
  const std::uint64_t conflicts = state.stats.pipeline.claim_conflict_count;
  const std::uint64_t barriers = state.stats.pipeline.barrier_count;
  const std::uint32_t sampled_runs = state.stats.pipeline.sampled_runs;
  const std::uint32_t clean_runs = state.stats.pipeline.clean_runs;
  const PreparationEvidenceSource preparation_evidence =
      state.stats.pipeline.preparation_evidence;
  const PublicationStats publication = state.stats.publication;
  state.stats = Stats{.backend = state.device->backend,
                      .graph_hash = state.publication->fingerprint.lo};
  state.stats.pipeline = PipelineStats{
      .step_count = state.logical_step_count,
      .resource_count = state.resources.size(),
      .barrier_count = barriers,
      .sealed_repetition_count = state.sealed_repetitions,
      .claim_conflict_count = conflicts,
      .failed_step_index = PipelineStats::no_failed_step,
      .status_entry_count = state.status_entry_count,
      .preparation_evidence = preparation_evidence,
      .prepared_template_count = state.plan.prepared_template_count,
      .prepared_command_count = state.plan.prepared_command_count,
      .sampled_runs = sampled_runs,
      .clean_runs = clean_runs,
  };
  state.stats.publication = publication;
}

Status pipeline_window_status(const PipelineWindows &windows,
                              const PipelineStep &step, const Status status,
                              ControlStats &stats) noexcept {
  const PipelineWindow *const descriptor = windows.find(step.window);
  if (descriptor == nullptr || status.reason() != Reason::BoundedCountInvalid) {
    return status;
  }
  stats.overflow_ordinal = descriptor->control.maximum;
  return status;
}

PipelineOutcome finish_accel_pipeline(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  PipelineOutcome result{};
  Stats stats = stats_from_evidence(state.device->backend, evidence.shared, 0u);
  stats.publication = state.stats.publication;
  stats.graph_hash = state.publication->fingerprint.lo;
  stats.pipeline = state.stats.pipeline;
  stats.pipeline.status_entry_count = evidence.status_entry_count;
  stats.pipeline.control_byte_count = evidence.control_byte_count;
  stats.pipeline.control_command_count = evidence.control_command_count;
  stats.pipeline.control_ns = evidence.control_ns;

  const PipelineEvidenceDecision decision = decide_ordinary_pipeline_evidence(
      {.total_steps = state.steps.size(),
       .active_steps = state.active_step_count,
       .generation = state.attempt.generation,
       .prior_submission = state.attempt.backend_submitted,
       .timed_metal = state.attempt.dispatch_timing &&
                      state.device->backend == Backend::Metal},
      evidence);
  result = decision.outcome;
  if (result.unknown_terminal()) {
    state.control_poisoned = true;
  }
  if (decision.control_valid) {
    const auto coordinate = [](const std::uint32_t value) noexcept {
      return value == node::accel::detail::PreparedPipelineNoStep
                 ? PipelineStats::no_coordinate
                 : static_cast<std::uint64_t>(value);
    };
    stats.pipeline.failed_outer_window =
        coordinate(evidence.control.failed_outer_window);
    stats.pipeline.failed_inner_iteration =
        coordinate(evidence.control.failed_inner_iteration);
    stats.pipeline.failed_nested_phase = decision.failed_nested_phase;
    stats.pipeline.executed_outer_window_count =
        evidence.control.executed_outer_window_count;
    stats.pipeline.skipped_outer_window_count =
        evidence.control.skipped_outer_window_count;
    stats.pipeline.executed_inner_iteration_count =
        evidence.control.executed_inner_iteration_count;
    stats.pipeline.skipped_inner_iteration_count =
        evidence.control.skipped_inner_iteration_count;
  }

  if (!state.window_rank.empty()) {
    ::rund::detail::counter::Accumulate(stats.control.iteration_count,
                                        state.window_rank[result.verified]);
  }
  if (!result.status && result.failed_step.has_value()) {
    result.status = pipeline_window_status(
        state.windows, state.steps[result.failed_step.value_or(0u)],
        result.status, stats.control);
  }
  capture_accel_pipeline_profile(
      state, evidence,
      decision.layout_valid && !result.unknown_terminal() &&
          (decision.control_valid || (result.status && !result.submitted())));
  if (!result.status &&
      !rebase_failed_pipeline_generation(state, result.submitted(),
                                         result.status.reason())) {
    result.status = Status::fail(Reason::PipelinePoisoned);
  }
  stats.pipeline.verified_step_count =
      logical_verified_steps(state, result.verified);
  stats.pipeline.failed_step_index =
      result.failed_step.has_value()
          ? logical_step_index(state, result.failed_step.value_or(0u))
          : PipelineStats::no_failed_step;
  state.attempt.verified = result.verified;
  state.attempt.failure_step_known = result.failed_step.has_value();
  state.stats = stats;
  result.writes_possible = result.submitted();
  return result;
}

} // namespace rund::compute::detail
