#include "result.hpp"

#include "../local.hpp"

#include "../../stats.hpp"
#include "../../status.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail {
namespace {

[[nodiscard]] const PipelineWindow *window(const PipelineState &state,
                                           const PipelineStep &step) noexcept {
  const std::size_t index = step.window;
  return index == 0u || index > state.windows.size()
             ? nullptr
             : &state.windows[index - 1u];
}

} // namespace

void reset_pipeline_stats(PipelineState &state) noexcept {
  const std::uint64_t conflicts = state.stats.pipeline.claim_conflict_count;
  const std::uint64_t barriers = state.stats.pipeline.barrier_count;
  const PublicationStats publication = state.stats.publication;
  state.stats = Stats{.backend = state.device->backend,
                      .graph_hash = state.fingerprint.lo};
  state.stats.pipeline = PipelineStats{
      .step_count = state.logical_step_count,
      .resource_count = state.resources.size(),
      .barrier_count = barriers,
      .claim_conflict_count = conflicts,
      .failed_step_index = PipelineStats::no_failed_step,
      .status_entry_count = state.status_entry_count,
  };
  state.stats.publication = publication;
}

Status pipeline_window_status(PipelineState &state, const PipelineStep &step,
                              const Status status, Stats &stats) noexcept {
  const PipelineWindow *const descriptor = window(state, step);
  if (descriptor == nullptr || status.reason() != Reason::BoundedCountInvalid) {
    return status;
  }
  stats.control.overflow_ordinal = descriptor->maximum;
  return status;
}

PipelineOutcome finish_accel_pipeline(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  PipelineOutcome result{};
  Stats stats = stats_from_evidence(state.device->backend, evidence.shared, 0u);
  stats.publication = state.stats.publication;
  stats.graph_hash = state.fingerprint.lo;
  stats.pipeline = state.stats.pipeline;
  stats.pipeline.status_entry_count = evidence.status_entry_count;
  stats.pipeline.control_byte_count = evidence.control_byte_count;
  stats.pipeline.control_command_count = evidence.control_command_count;
  stats.pipeline.control_ns = evidence.control_ns;

  result.status = evidence.check.ok
                      ? Status::success()
                      : Status::fail(project_reason(evidence.check.reason,
                                                    Reason::BackendFailed));
  const bool layout_valid =
      state.active_step_count == evidence.active_step_count;
  if (!layout_valid) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }
  result.submitted = evidence.submitted || state.backend_submitted ||
                     stats.command_submits != 0u;
  if (result.submitted && stats.command_submits != 1u) {
    result.status = Status::fail(Reason::CompletionInvalid);
    stats.command_submits = 1u;
  } else if (!result.submitted && stats.command_submits != 0u) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }

  const bool control_valid =
      evidence.control_observed && evidence.control_valid &&
      evidence.control_byte_count ==
          node::accel::detail::PreparedPipelineControlBytes &&
      node::accel::detail::PreparedPipelineGenerationMatches(
          evidence.control, state.generation + 1u);
  if (evidence.control_observed && !control_valid) {
    result.status = Status::fail(Reason::CompletionInvalid);
  } else if (control_valid) {
    result.verified = std::min<std::size_t>(evidence.control.verified_prefix,
                                            state.steps.size());
    result.failure_step_known =
        !result.status &&
        evidence.control.failed_step !=
            node::accel::detail::PreparedPipelineNoStep &&
        evidence.control.failed_step < state.steps.size();
    if (result.failure_step_known) {
      result.failed_step = evidence.control.failed_step;
    }
    result.publication_suppressed = evidence.control.reason != 0u;
  } else if (result.status && !result.submitted) {
    result.verified = state.steps.size();
  }

  if (!state.window_rank.empty()) {
    ::rund::detail::counter::Accumulate(stats.control.iteration_count,
                                        state.window_rank[result.verified]);
  }
  if (!result.status && result.failure_step_known) {
    result.status = pipeline_window_status(
        state, state.steps[result.failed_step], result.status, stats);
  }
  capture_accel_pipeline_profile(
      state, evidence,
      layout_valid && (control_valid || (result.status && !result.submitted)));
  if (!result.status && !rebase_failed_pipeline_generation(
                            state, result.submitted, result.status.reason())) {
    result.status = Status::fail(Reason::PipelinePoisoned);
  }
  stats.pipeline.verified_step_count =
      logical_verified_steps(state, result.verified);
  stats.pipeline.failed_step_index =
      result.failure_step_known ? logical_step_index(state, result.failed_step)
                                : PipelineStats::no_failed_step;
  state.verified = result.verified;
  state.failure_step_known = result.failure_step_known;
  state.stats = stats;
  result.writes_possible = result.submitted;
  return result;
}

} // namespace rund::compute::detail
