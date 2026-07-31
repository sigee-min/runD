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

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineOutcome run_cpu(PipelineState &state) {
  PipelineOutcome outcome{.publication_suppressed = true};
  reset_cpu_resident(state);
  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    PipelineStep &step = state.steps[index];
    begin_pipeline_profile_step(state, index);
    const auto finish_step = [&](const bool executed) noexcept {
      capture_cpu_pipeline_step(state, index, executed);
      finish_pipeline_profile_step(state, index);
    };
    bool active = true;
    const Status window_ready =
        prepare_cpu_pipeline_window(state, index, active);
    if (!window_ready) {
      finish_step(false);
      outcome.status =
          pipeline_window_status(state, step, window_ready, state.stats);
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    if (!active) {
      finish_step(false);
      ++outcome.verified;
      continue;
    }
    outcome.writes_possible = outcome.writes_possible || step.writes;
    const std::shared_ptr<JobState> &job =
        state.transactional && state.parity != 0u ? step.alternate_job
                                                  : step.job;
    const Status gathered = gather_cpu_pipeline_views(job);
    if (!gathered) {
      finish_step(false);
      outcome.status = gathered;
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    const Status ran = run_pipeline_job(job);
    if (!ran) {
      const Status consumed = consume_cpu_pipeline_step(state, index, ran);
      finish_step(true);
      outcome.status = pipeline_window_status(
          state, step, consumed ? ran : consumed, state.stats);
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    const Status published = publish_cpu_pipeline_views(job);
    if (!published) {
      finish_step(true);
      outcome.status = published;
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    const Status consumed = consume_cpu_pipeline_step(state, index);
    if (!consumed) {
      finish_step(true);
      outcome.status = consumed;
      outcome.failed_step = index;
      outcome.failure_step_known = true;
      return outcome;
    }
    finish_step(true);
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
      state.transactional && state.parity != 0u ? state.alternate_prepared
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
  return state->stats;
}

graph::Fingerprint
pipeline_fingerprint(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr ? graph::Fingerprint{} : state->fingerprint;
}

std::uint64_t
pipeline_generation(const std::shared_ptr<PipelineState> &state) noexcept {
  if (state == nullptr) {
    return 0u;
  }
  std::lock_guard lock{state->gate};
  return state->generation;
}

} // namespace rund::compute::detail
