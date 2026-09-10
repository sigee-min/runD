#include "../local.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../backend.hpp"
#include "../../device/info.hpp"
#include "../../job/local.hpp"
#include "../../stats.hpp"
#include "../../status.hpp"
#include "../../terminal.hpp"
#include "../claim.hpp"
#include "../run/memory.hpp"
#include "../run/result.hpp"

#include <mutex>

namespace rund::compute::detail {
namespace {

[[nodiscard]] TerminalObservation
observe_pipeline_terminal_locked(PipelineState &state, const Status status,
                                 const std::uint64_t frame_bytes,
                                 const bool capture_profile) noexcept {
  ::rund::detail::counter::Release(state.frame_current, frame_bytes);
  if (!capture_profile) {
    return TerminalObservationAccess::from_stats(status, state.stats);
  }
  if (state.publication != nullptr) {
    std::lock_guard publication_lock{state.publication->gate};
    synchronize_pipeline_observation_epoch(state, *state.publication);
    state.stats.publication.generation = state.publication->generation;
  }
  const MemoryStats memory = pipeline_memory_view_locked(state).summary;
  return TerminalObservationAccess::from_profile(
      status, device_info_owner(state.device), state.stats, memory);
}

[[nodiscard]] Status cancel_pipeline_locked(PipelineState &state) noexcept {
  if (state.phase != PipelinePhase::Running) {
    return state.failure == Reason::Cancelled ? Status::fail(Reason::Cancelled)
           : state.phase == PipelinePhase::Poisoned
               ? Status::fail(Reason::PipelinePoisoned)
               : Status::fail(Reason::AlreadyCompleted);
  }
  publish_pipeline_terminal(
      state,
      PipelineTerminal{
          .reason = Reason::Cancelled,
          .verified = state.attempt.verified,
          .failed_step = state.attempt.verified,
          .failure_step_known = state.attempt.failure_step_known,
          .writes_possible = state.attempt.writes_possible || state.attempt.backend_submitted,
          .publication_suppressed = state.device->backend == Backend::Cpu ||
                                    !state.attempt.backend_submitted});
  return Status::fail(Reason::Cancelled);
}

[[nodiscard]] Status fail_pipeline_locked(PipelineState &state,
                                          const Status failure) noexcept {
  if (state.phase == PipelinePhase::Running) {
    publish_pipeline_terminal(
        state,
        PipelineTerminal{
            .reason = failure.reason(),
            .verified = state.attempt.verified,
            .failed_step = state.attempt.verified,
            .failure_step_known = state.attempt.failure_step_known,
            .writes_possible = state.attempt.writes_possible || state.attempt.backend_submitted,
            .publication_suppressed = state.device->backend == Backend::Cpu ||
                                      !state.attempt.backend_submitted});
  }
  return Status::fail(failure.reason());
}

[[nodiscard]] Status complete_cpu_pipeline_locked(PipelineState &state) {
  if (state.phase != PipelinePhase::Running ||
      state.attempt.verified != state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.stats.command_submits = 0u;
  const Status published = publish_cpu_pipeline(state);
  if (!published) {
    publish_pipeline_terminal(
        state,
        PipelineTerminal{.reason = published.reason(),
                         .verified = state.attempt.verified,
                         .failed_step =
                             state.steps.empty() ? 0u : state.steps.size() - 1u,
                         .failure_step_known = !state.steps.empty(),
                         .writes_possible = true,
                         .publication_suppressed = false});
    return published;
  }
  publish_pipeline_terminal(state, PipelineTerminal{});
  return Status::success();
}

[[nodiscard]] Status finish_pipeline_locked(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  if (state.phase != PipelinePhase::Running) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr) {
    publish_pipeline_terminal(
        state,
        PipelineTerminal{.reason = Reason::AccelProgramInvalid,
                         .verified = state.attempt.verified,
                         .failed_step = state.attempt.verified,
                         .failure_step_known = state.attempt.failure_step_known,
                         .writes_possible =
                             state.attempt.writes_possible || state.attempt.backend_submitted,
                         .publication_suppressed = !state.attempt.backend_submitted});
    return Status::fail(Reason::AccelProgramInvalid);
  }
  const PipelineOutcome outcome = finish_accel_pipeline(state, evidence);
  if (!outcome.status) {
    publish_pipeline_terminal(
        state,
        PipelineTerminal{
            .reason = outcome.status.reason(),
            .verified = outcome.verified,
            .failed_step = outcome.failed_step.value_or(0u),
            .failure_step_known = outcome.failed_step.has_value(),
            .writes_possible = outcome.writes_possible || outcome.submitted(),
            .publication_suppressed = outcome.publication_suppressed});
    return outcome.status;
  }
  publish_pipeline_terminal(state, PipelineTerminal{});
  return Status::success();
}

} // namespace

Status cancel_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return cancel_pipeline_locked(*state);
}

TerminalObservation
cancel_pipeline_terminal(const std::shared_ptr<PipelineState> &state,
                         const std::uint64_t frame_bytes,
                         const bool capture_profile) noexcept {
  if (!valid_pipeline(state)) {
    return TerminalObservationAccess::from_stats(
        Status::fail(Reason::PipelineInvalid));
  }
  std::lock_guard lock{state->gate};
  const Status status = cancel_pipeline_locked(*state);
  return observe_pipeline_terminal_locked(*state, status, frame_bytes,
                                          capture_profile);
}

Status fail_pipeline(const std::shared_ptr<PipelineState> &state,
                     const Status failure) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return fail_pipeline_locked(*state, failure);
}

TerminalObservation
fail_pipeline_terminal(const std::shared_ptr<PipelineState> &state,
                       const Status failure, const std::uint64_t frame_bytes,
                       const bool capture_profile) noexcept {
  if (!valid_pipeline(state)) {
    return TerminalObservationAccess::from_stats(
        Status::fail(Reason::PipelineInvalid));
  }
  std::lock_guard lock{state->gate};
  const Status status = fail_pipeline_locked(*state, failure);
  return observe_pipeline_terminal_locked(*state, status, frame_bytes,
                                          capture_profile);
}

TerminalObservation
complete_cpu_pipeline_terminal(const std::shared_ptr<PipelineState> &state,
                               const std::uint64_t frame_bytes,
                               const bool capture_profile) noexcept {
  if (!valid_pipeline(state)) {
    return TerminalObservationAccess::from_stats(
        Status::fail(Reason::PipelineInvalid));
  }
  std::lock_guard lock{state->gate};
  const Status status = complete_cpu_pipeline_locked(*state);
  return observe_pipeline_terminal_locked(*state, status, frame_bytes,
                                          capture_profile);
}

Status finish_pipeline_on(
    const std::shared_ptr<PipelineState> &state,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  return finish_pipeline_locked(*state, evidence);
}

TerminalObservation finish_pipeline_terminal_on(
    const std::shared_ptr<PipelineState> &state,
    node::accel::detail::PreparedPipelineEvidence &&evidence,
    const std::uint64_t frame_bytes, const bool capture_profile) noexcept {
  if (!valid_pipeline(state)) {
    return TerminalObservationAccess::from_stats(
        Status::fail(Reason::PipelineInvalid));
  }
  std::lock_guard lock{state->gate};
  const Status status = finish_pipeline_locked(*state, evidence);
  return observe_pipeline_terminal_locked(*state, status, frame_bytes,
                                          capture_profile);
}

} // namespace rund::compute::detail
