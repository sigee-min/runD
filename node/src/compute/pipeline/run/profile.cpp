#include "clock.hpp"

#include "../local.hpp"
#include "../state.hpp"

#include "../../../accel/kernel/prepared.hpp"
#include "../../cpu/graph.hpp"
#include "../../job/local.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstddef>

namespace rund::compute::detail {
namespace {

void accumulate(PipelineStepStats &target,
                const PipelineStepStats &source) noexcept {
  if (target.sample_count == 0u) {
    target = source;
    return;
  }
  ::rund::detail::counter::Accumulate(target.sample_count, source.sample_count);
  ::rund::detail::counter::Accumulate(target.original_dispatches,
                                      source.original_dispatches);
  ::rund::detail::counter::Accumulate(target.final_dispatches,
                                      source.final_dispatches);
  ::rund::detail::counter::Accumulate(target.barrier_count,
                                      source.barrier_count);
  target.worker_count = std::max(target.worker_count, source.worker_count);
  target.participating_workers =
      std::max(target.participating_workers, source.participating_workers);
  ::rund::detail::counter::Accumulate(target.tile_count, source.tile_count);
  target.tile_size = std::max(target.tile_size, source.tile_size);
  ::rund::detail::counter::Accumulate(target.vector_chunks,
                                      source.vector_chunks);
  ::rund::detail::counter::Accumulate(target.tail_chunks, source.tail_chunks);
  ::rund::detail::counter::Accumulate(target.workgroup_count,
                                      source.workgroup_count);
  ::rund::detail::counter::Accumulate(target.work_item_count,
                                      source.work_item_count);
  ::rund::detail::counter::Accumulate(target.control.generated_item_count,
                                      source.control.generated_item_count);
  ::rund::detail::counter::Accumulate(target.control.generated_capacity,
                                      source.control.generated_capacity);
  ::rund::detail::counter::Accumulate(target.control.indirect_dispatch_count,
                                      source.control.indirect_dispatch_count);
  ::rund::detail::counter::Accumulate(target.control.indirect_work_item_count,
                                      source.control.indirect_work_item_count);
  ::rund::detail::counter::Accumulate(target.control.iteration_count,
                                      source.control.iteration_count);
  ::rund::detail::counter::Accumulate(target.control.skipped_iteration_count,
                                      source.control.skipped_iteration_count);
  ::rund::detail::counter::Accumulate(target.control.conflict_count,
                                      source.control.conflict_count);
  target.control.overflow_ordinal = std::min(target.control.overflow_ordinal,
                                             source.control.overflow_ordinal);
}

} // namespace

void reset_pipeline_profile(PipelineState &state) noexcept {
  if (state.profile == nullptr) {
    return;
  }
  PipelineProfileState &profile = *state.profile;
  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    const PipelineStep &step = state.steps[index];
    PipelineStepProfile row{
        .index = step.logical_step,
        .iteration = step.iteration,
        .iteration_bound = step.iteration_bound,
        .program = step.program == nullptr
                       ? graph::Fingerprint{}
                       : step.program->graph_info.fingerprint,
    };
    const PipelineWindow *const nested =
        step.window != 0u && step.window <= state.windows.size() &&
                state.windows[step.window - 1u].nested()
            ? &state.windows[step.window - 1u]
            : nullptr;
    if (nested != nullptr) {
      row.outer_window_bound = nested->nested_shape.outer_bound();
      row.inner_iteration_bound = nested->nested_shape.inner_bound();
    }
    if (step.route == PipelineRoute::NestedSeed) {
      row.outer_window = step.iteration;
      row.nested_phase = PipelineNestedPhase::Seed;
    } else if (step.route == PipelineRoute::NestedAction) {
      row.inner_iteration = step.iteration;
      row.nested_phase = PipelineNestedPhase::Action;
    } else if (step.route == PipelineRoute::NestedFold) {
      row.nested_phase = PipelineNestedPhase::Fold;
    }
    profile.steps[index] = row;
  }
  std::fill(profile.started.begin(), profile.started.end(), false);
  profile.instrumentation_command_count = 0u;
  profile.instrumentation_byte_count = 0u;
}

void begin_pipeline_profile_step(PipelineState &state,
                                 const std::size_t index) noexcept {
  if (state.profile == nullptr || index >= state.steps.size()) {
    return;
  }
  state.profile->started_ns[index] = pipeline_clock();
  state.profile->started[index] = true;
}

void finish_pipeline_profile_step(PipelineState &state,
                                  const std::size_t index) noexcept {
  if (state.profile == nullptr || index >= state.steps.size() ||
      !state.profile->started[index]) {
    return;
  }
  const std::uint64_t finished = pipeline_clock();
  const std::uint64_t started = state.profile->started_ns[index];
  state.profile->started[index] = false;
  StepTiming &timing = state.profile->steps[index].timing;
  ::rund::detail::counter::Accumulate(
      timing.duration_ns, finished >= started ? finished - started : 0u);
  ::rund::detail::counter::Accumulate(timing.sample_count, 1u);
  timing.clock = StepClock::HostSteady;
  timing.relation = StepTimingRelation::Exclusive;
}

void capture_cpu_pipeline_step(PipelineState &state, const std::size_t index,
                               const bool executed) noexcept {
  if (state.profile == nullptr || index >= state.steps.size() || !executed) {
    return;
  }
  PipelineStepStats stats{
      .sample_count = 1u,
      .barrier_count =
          index < state.barriers.size() && state.barriers[index] != 0u ? 1u
                                                                       : 0u,
  };
  const PipelineStep &step = state.steps[index];
  if (step.program != nullptr && !step.program->empty()) {
    const std::shared_ptr<JobState> &job =
        state.transactional && state.attempt_parity != 0u ? step.alternate_job
                                                          : step.job;
    if (job == nullptr || job->cpu == nullptr || job->cpu->graph == nullptr) {
      return;
    }
    const Stats &run = job->cpu->stats;
    stats.original_dispatches =
        step.program->cpu_graph != nullptr &&
                step.program->cpu_graph->runtime != nullptr
            ? step.program->cpu_graph->runtime->steps.size()
            : 0u;
    stats.final_dispatches = run.dispatches;
    stats.worker_count = run.worker_count;
    stats.participating_workers = run.participating_workers;
    stats.tile_count = run.tile_count;
    stats.tile_size = run.tile_size;
    stats.vector_chunks = run.vector_chunks;
    stats.tail_chunks = run.tail_chunks;
    stats.control = run.control;
    stats.control.conflict_count = job->cpu->graph->conflict_count;
    stats.control.overflow_ordinal = job->cpu->graph->overflow_ordinal;
  }
  accumulate(state.profile->steps[index].execution, stats);
}

void capture_accel_pipeline_profile(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence,
    const bool row_identity_valid) noexcept {
  if (state.profile == nullptr) {
    return;
  }
  PipelineProfileState &profile = *state.profile;
  profile.instrumentation_command_count =
      evidence.profile.instrumentation_command_count;
  profile.instrumentation_byte_count =
      evidence.profile.instrumentation_byte_count;
  if (!row_identity_valid) {
    return;
  }
  if (state.active_step_count == 0u && evidence.check.ok &&
      !evidence.submitted) {
    for (std::size_t index = 0u; index < state.steps.size(); ++index) {
      profile.steps[index].execution = PipelineStepStats{
          .sample_count = 1u,
          .barrier_count =
              index < state.barriers.size() && state.barriers[index] != 0u ? 1u
                                                                           : 0u,
      };
    }
    return;
  }
  if (evidence.active_step_count != state.active_step_count ||
      !evidence.profile.observed ||
      evidence.profile.steps.size() != state.steps.size()) {
    return;
  }
  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    const node::accel::detail::PreparedPipelineStepEvidence &source =
        evidence.profile.steps[index];
    PipelineStepProfile &target = profile.steps[index];
    if (source.work_sample_count != 0u) {
      target.execution = PipelineStepStats{
          .sample_count = source.work_sample_count,
          .original_dispatches = source.original_dispatch_count,
          .final_dispatches = source.physical_dispatch_count,
          .barrier_count =
              index < state.barriers.size() && state.barriers[index] != 0u ? 1u
                                                                           : 0u,
          .workgroup_count = source.workgroup_count,
          .work_item_count = source.work_item_count,
          .control = source.control,
      };
    }
    if (source.timing_sample_count != 0u &&
        source.clock ==
            node::accel::detail::PreparedPipelineStepClock::Device &&
        source.relation ==
            node::accel::detail::PreparedPipelineStepTimingRelation::
                NonAdditive) {
      target.timing = StepTiming{.duration_ns = source.duration_ns,
                                 .sample_count = source.timing_sample_count,
                                 .clock = StepClock::Device,
                                 .relation = StepTimingRelation::NonAdditive};
    }
  }
}

} // namespace rund::compute::detail
