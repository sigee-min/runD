#include <rund/compute/pipeline.hpp>

#include "../backend.hpp"
#include "../device/residency/pool.hpp"
#include "../status.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "run/internal.hpp"
#include "state.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {
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

Status
begin_pipeline_samples(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == PipelinePhase::Running) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->phase == PipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  state->stats.pipeline.sampled_runs = 0u;
  state->stats.pipeline.clean_runs = 0u;
  state->samples = PipelineState::SampleState::Clean;
  return Status::success();
}

Status
end_pipeline_samples(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == PipelinePhase::Running) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->phase == PipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (state->samples == PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileInvalid);
  }
  state->samples = PipelineState::SampleState::Inactive;
  return Status::success();
}

Status run_pipeline(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->device->backend != Backend::Cpu && state->device->ops != nullptr &&
      state->device->ops->virtual_execution.run_service_free_direct_product !=
          nullptr) {
    bool selected = false;
    const Status service_free =
        state->device->ops->virtual_execution.run_service_free_direct_product(
            state, selected);
    if (selected) {
      return service_free;
    }
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  constexpr PipelineClaimAuthority claim_authority =
      PipelineClaimAuthority::Shared;
  const Status started = start_pipeline(*state, claim_authority);
  if (!started) {
    return started;
  }
  const PipelineOutcome executed =
      state->device->backend == Backend::Cpu
          ? run_cpu(*state, state->steps.size())
          : run_accel(*state,
                      node::accel::detail::PipelineSubmitMode::Standard);
  if (executed.status) {
    publish_pipeline_terminal(*state, PipelineTerminal{}, claim_authority);
  } else {
    publish_pipeline_terminal(
        *state,
        PipelineTerminal{
            .reason = executed.status.reason(),
            .verified = executed.verified,
            .failed_step = executed.failed_step.value_or(0u),
            .failure_step_known = executed.failed_step.has_value(),
            .writes_possible = executed.writes_possible || executed.submitted(),
            .publication_suppressed = executed.publication_suppressed},
        claim_authority);
  }
  return executed.status;
}

void submit_residency_pipeline_lease(
    const std::shared_ptr<PipelineState> &state,
    const residency::EpochLease lease, ResidencyPipelineSubmission &submission,
    const ResidencyPipelineCompletion completion, void *const user,
    const bool defer_generation, const std::uint64_t control_generation,
    const std::uint8_t control_parity,
    const std::uint64_t publication_generation,
    const std::uint8_t publication_parity) noexcept {
  ResidencyBackendSelection selection{};
  const Status admitted = prepare_residency_pipeline_submission(
      state, lease, submission, completion, user, selection, defer_generation,
      control_generation, control_parity, publication_generation,
      publication_parity);
  if (!admitted) {
    fail_residency_pipeline_submission(submission, admitted);
    return;
  }
  const std::span<const std::uint32_t> locals{submission.locals.data(),
                                              submission.issued_steps};
  if (selection.ops->residency.submit_residency_pipeline == nullptr) {
    fail_residency_pipeline_submission(submission,
                                       Status::fail(Reason::PipelineInvalid));
    return;
  }
  const rund::AccelCheck submitted =
      selection.ops->residency.submit_residency_pipeline(
          *state->device, *selection.prepared, locals, state,
          complete_residency_pipeline_submission, &submission);
  if (submitted.ok) {
    ResidencySubmissionPhase expected = ResidencySubmissionPhase::Submitting;
    static_cast<void>(submission.phase.compare_exchange_strong(
        expected, ResidencySubmissionPhase::Submitted,
        std::memory_order_acq_rel, std::memory_order_acquire));
    return;
  }
  fail_residency_pipeline_submission(
      submission,
      Status::fail(project_reason(submitted.reason, Reason::BackendFailed)));
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
