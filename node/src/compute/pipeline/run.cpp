#include <rund/compute/pipeline.hpp>

#include "../../accel/kernel/prepared.hpp"
#include "../backend.hpp"
#include "../job/local.hpp"
#include "../memory/cpu.hpp"
#include "../memory/local.hpp"
#include "../stats.hpp"
#include "../status.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "run/clock.hpp"
#include "run/result.hpp"
#include "state.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>

#include <accel/kernel/evidence.hpp>
#include <rund/counter.hpp>

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineOutcome run_cpu(PipelineState &state) {
  PipelineOutcome outcome{.publication_suppressed = true};
  reset_cpu_resident(state);
  const auto execute = [&](const std::size_t index) {
    if (index >= state.steps.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    PipelineStep &step = state.steps[index];
    begin_pipeline_profile_step(state, index);
    const auto finish_step = [&](const bool executed) noexcept {
      capture_cpu_pipeline_step(state, index, executed);
      finish_pipeline_profile_step(state, index);
    };
    outcome.writes_possible = outcome.writes_possible || step.writes;
    const std::shared_ptr<JobState> &job =
        state.transactional && state.attempt_parity != 0u ? step.alternate_job
                                                          : step.job;
    const Status gathered = gather_cpu_pipeline_views(job);
    if (!gathered) {
      finish_step(false);
      return gathered;
    }
    const Status ran = run_pipeline_job(job);
    if (!ran) {
      const Status consumed = consume_cpu_pipeline_step(state, index, ran);
      finish_step(true);
      return consumed ? ran : consumed;
    }
    const Status published = publish_cpu_pipeline_views(job);
    if (!published) {
      finish_step(true);
      return published;
    }
    const Status consumed = consume_cpu_pipeline_step(state, index);
    finish_step(true);
    return consumed;
  };

  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    PipelineStep &step = state.steps[index];
    if (step.route == PipelineRoute::NestedSeed) {
      if (step.window == 0u || step.window > state.windows.size()) {
        outcome.status = Status::fail(Reason::PipelineInvalid);
        outcome.failed_step = index;
        outcome.failure_step_known = true;
        return outcome;
      }
      PipelineWindow &nested = state.windows[step.window - 1u];
      const node::accel::detail::NestedTemplateShape &shape =
          nested.nested_shape;
      node::accel::detail::NestedTemplateShape expected_shape{};
      if (!nested.nested() || index != shape.first() ||
          nested.recurrent_output_count == 0u ||
          shape.end() > state.steps.size() ||
          !node::accel::detail::ProveNestedTemplateShape(
              shape.first(), nested.control.maximum, nested.control.tile,
              shape.inner_bound(), expected_shape) ||
          expected_shape != shape) {
        outcome.status = Status::fail(Reason::PipelineInvalid);
        outcome.failed_step = index;
        outcome.failure_step_known = true;
        return outcome;
      }

      std::size_t executed_outer = 0u;
      for (std::size_t outer = 0u; outer < shape.outer_bound(); ++outer) {
        PipelineStep occurrence = state.steps[shape.fold_first()];
        occurrence.iteration = static_cast<std::uint32_t>(outer);
        occurrence.iteration_bound = shape.outer_bound();
        bool active = true;
        const Status ready =
            prepare_cpu_pipeline_window(state, occurrence, active);
        if (!ready) {
          outcome.status =
              pipeline_window_status(state, occurrence, ready, state.stats);
          outcome.failed_step = shape.first();
          outcome.failure_step_known = true;
          state.stats.pipeline.failed_outer_window = outer;
          state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Seed;
          nested.stopped = true;
          return outcome;
        }
        if (!active) {
          continue;
        }

        const std::size_t seed_index = shape.seed_first() + outer;
        Status status = execute(seed_index);
        if (!status) {
          outcome.status = status;
          outcome.failed_step = seed_index;
          outcome.failure_step_known = true;
          state.stats.pipeline.failed_outer_window = outer;
          state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Seed;
          nested.stopped = true;
          return outcome;
        }
        for (std::size_t inner = 0u; inner < shape.action_count(); ++inner) {
          const std::size_t action_index = shape.action_first() + inner;
          status = execute(action_index);
          if (!status) {
            outcome.status = status;
            outcome.failed_step = action_index;
            outcome.failure_step_known = true;
            state.stats.pipeline.failed_outer_window = outer;
            state.stats.pipeline.failed_inner_iteration = inner;
            state.stats.pipeline.failed_nested_phase =
                PipelineNestedPhase::Action;
            nested.stopped = true;
            return outcome;
          }
          ++state.stats.pipeline.executed_inner_iteration_count;
        }
        std::uint32_t fold_route = 0u;
        if (!shape.fold_route_for_outer(static_cast<std::uint32_t>(outer),
                                        fold_route)) {
          outcome.status = Status::fail(Reason::PipelineInvalid);
          outcome.failed_step = shape.fold_first();
          outcome.failure_step_known = true;
          return outcome;
        }
        const std::size_t fold_index = shape.fold_first() + fold_route;
        status = execute(fold_index);
        if (!status) {
          outcome.status = status;
          outcome.failed_step = fold_index;
          outcome.failure_step_known = true;
          state.stats.pipeline.failed_outer_window = outer;
          state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Fold;
          nested.stopped = true;
          return outcome;
        }
        bool window_wrote = false;
        status = publish_cpu_pipeline_window(state, step.window, outer,
                                             window_wrote);
        outcome.writes_possible = outcome.writes_possible || window_wrote;
        if (!status) {
          outcome.status = status;
          outcome.failed_step = fold_index;
          outcome.failure_step_known = true;
          state.stats.pipeline.failed_outer_window = outer;
          state.stats.pipeline.failed_nested_phase = PipelineNestedPhase::Fold;
          nested.stopped = true;
          return outcome;
        }
        nested.current =
            fold_route == 1u ? PipelineWindow::second : PipelineWindow::first;
        ++executed_outer;
        ++state.stats.pipeline.executed_outer_window_count;
        ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                            1u);
      }
      const std::size_t skipped = shape.seed_count() - executed_outer;
      ::rund::detail::counter::Accumulate(
          state.stats.pipeline.skipped_outer_window_count,
          static_cast<std::uint64_t>(skipped));
      ::rund::detail::counter::Accumulate(
          state.stats.pipeline.skipped_inner_iteration_count,
          ::rund::detail::counter::SaturatingMultiply(
              static_cast<std::uint64_t>(skipped),
              static_cast<std::uint64_t>(shape.action_count())));
      outcome.verified = shape.end();
      index = shape.end() - 1u;
      continue;
    }
    if (step.route != PipelineRoute::Ordinary) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    bool active = true;
    const Status window_ready =
        prepare_cpu_pipeline_window(state, index, active);
    if (!window_ready) {
      outcome.status =
          pipeline_window_status(state, step, window_ready, state.stats);
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    if (!active) {
      ++outcome.verified;
      continue;
    }
    const Status executed = execute(index);
    if (!executed) {
      outcome.status =
          pipeline_window_status(state, step, executed, state.stats);
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    ++outcome.verified;
  }
  const Status published = publish_cpu_pipeline(state);
  if (!published) {
    outcome.status = published;
    outcome.failed_step = state.steps.empty() ? 0u : state.steps.size() - 1u;
    outcome.failure_step_known = !state.steps.empty();
    outcome.writes_possible = true;
    outcome.publication_suppressed = false;
    return outcome;
  }
  state.stats.command_submits = 0u;
  state.stats.pipeline.verified_step_count =
      logical_verified_steps(state, outcome.verified);
  state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
  return outcome;
}

[[nodiscard]] PipelineOutcome run_accel(PipelineState &state) {
  PipelineOutcome outcome{};
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr || ops->run_pipeline == nullptr) {
    outcome.status = Status::fail(Reason::AccelProgramInvalid);
    return outcome;
  }
  const std::size_t active = state.active_step_count;
  if (active == 0u) {
    node::accel::detail::PreparedPipelineEvidence empty{};
    empty.check = {true, "ok"};
    empty.shared.backend = state.device->backend == Backend::Metal
                               ? rund::AccelApi::Metal
                               : rund::AccelApi::Vulkan;
    empty.shared.ok = true;
    empty.shared.reason = "ok";
    return finish_accel_pipeline(state, empty);
  }
  const node::accel::detail::PreparedKernelPipeline &prepared =
      state.transactional && state.attempt_parity != 0u
          ? state.alternate_prepared
          : state.prepared;
  if (!prepared.ok) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
    return outcome;
  }

  return finish_accel_pipeline(state,
                               ops->run_pipeline(*state.device, prepared));
}

} // namespace

bool valid_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  return state != nullptr && state->device != nullptr && !state->steps.empty();
}

bool poisoned_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (state == nullptr) {
    return false;
  }
  std::lock_guard lock{state->gate};
  return state->phase == PipelinePhase::Poisoned;
}

Status run_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  const Status started = start_pipeline(*state);
  if (!started) {
    return started;
  }
  const PipelineOutcome executed = state->device->backend == Backend::Cpu
                                       ? run_cpu(*state)
                                       : run_accel(*state);
  if (executed.status) {
    publish_pipeline_terminal(*state, PipelineTerminal{});
  } else {
    publish_pipeline_terminal(
        *state,
        PipelineTerminal{
            .reason = executed.status.reason(),
            .verified = executed.verified,
            .failed_step = executed.failed_step,
            .failure_step_known = executed.failure_step_known,
            .writes_possible = executed.writes_possible || executed.submitted,
            .publication_suppressed = executed.publication_suppressed});
  }
  return executed.status;
}

Stats pipeline_stats(const std::shared_ptr<PipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  if (state->publication != nullptr) {
    std::lock_guard publication_lock{state->publication->gate};
    synchronize_pipeline_observation_epoch(*state, *state->publication);
    state->stats.publication.generation = state->publication->generation;
  }
  return state->stats;
}

CheckpointStats pipeline_checkpoint_stats(
    const std::shared_ptr<PipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return state->checkpoint_stats;
}

graph::Fingerprint
pipeline_fingerprint(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr || state->publication == nullptr
             ? graph::Fingerprint{}
             : state->publication->fingerprint;
}

std::uint64_t
pipeline_generation(const std::shared_ptr<PipelineState> &state) noexcept {
  if (state == nullptr) {
    return 0u;
  }
  std::lock_guard lock{state->gate};
  if (state->publication == nullptr) {
    return 0u;
  }
  std::lock_guard publication_lock{state->publication->gate};
  return state->publication->generation;
}

} // namespace rund::compute::detail
