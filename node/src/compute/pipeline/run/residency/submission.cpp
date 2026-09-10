#include "../internal.hpp"

#include "../../../backend.hpp"
#include "../../claim.hpp"
#include "../../execution/prepare.hpp"
#include "../../execution/submit.hpp"
#include "../../state.hpp"

#include <accel/kernel/evidence.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {

Status prepare_residency_pipeline_submission(
    const std::shared_ptr<PipelineState> &state,
    const residency::EpochLease lease, ResidencyPipelineSubmission &submission,
    const ResidencyPipelineCompletion completion, void *const user,
    ResidencyBackendSelection &selection, const bool defer_generation,
    const std::uint64_t control_generation, const std::uint8_t control_parity,
    const std::uint64_t publication_generation,
    const std::uint8_t publication_parity) noexcept {
  selection = {};
  if (completion == nullptr || user == nullptr || submission.active()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  submission.pipeline = state;
  submission.defer_generation = defer_generation;
  submission.control_generation = control_generation;
  submission.control_parity = control_parity;
  submission.publication_generation = publication_generation;
  submission.publication_parity = publication_parity;
  submission.completion = completion;
  submission.user = user;
  submission.phase.store(ResidencySubmissionPhase::Submitting,
                         std::memory_order_release);
  const Status projected = residency_pipeline_locals(
      state, lease, std::span<std::uint32_t>{submission.locals},
      submission.issued_steps);
  if (!projected) {
    return projected;
  }
  const std::span<const std::uint32_t> locals{submission.locals.data(),
                                              submission.issued_steps};
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->device->backend == Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  constexpr PipelineClaimAuthority authority =
      PipelineClaimAuthority::PrivateResidency;
  const Status started =
      start_pipeline(*state, authority, control_generation, control_parity);
  if (!started) {
    return started;
  }
  PipelineOutcome prepared_outcome =
      prepare_residency_pipeline_execution(*state, locals);
  submission.writes_possible = prepared_outcome.writes_possible;
  if (!prepared_outcome.status) {
    prepared_outcome.defer_generation = defer_generation;
    if (defer_generation) {
      static_cast<void>(publish_residency_pipeline_outcome(
          *state, submission.issued_steps, prepared_outcome, defer_generation,
          publication_generation, publication_parity));
    } else {
      static_cast<void>(publish_residency_pipeline_execution(
          *state, submission.issued_steps, prepared_outcome));
    }
    return prepared_outcome.status;
  }
  selection.ops = state->device->ops;
  selection.prepared = state->transactional && state->attempt.parity != 0u
                           ? &state->alternate_prepared
                           : &state->prepared;
  if (selection.ops == nullptr || selection.prepared == nullptr ||
      !selection.prepared->ok) {
    PipelineOutcome rejected{.status = Status::fail(Reason::PipelineInvalid),
                             .publication_suppressed = true};
    rejected.defer_generation = defer_generation;
    if (defer_generation) {
      static_cast<void>(publish_residency_pipeline_outcome(
          *state, submission.issued_steps, rejected, defer_generation,
          publication_generation, publication_parity));
    } else {
      static_cast<void>(publish_residency_pipeline_execution(
          *state, submission.issued_steps, rejected));
    }
    selection = {};
    return rejected.status;
  }
  return Status::success();
}

void fail_residency_pipeline_submission(ResidencyPipelineSubmission &submission,
                                        const Status failure) noexcept {
  ResidencySubmissionPhase expected = ResidencySubmissionPhase::Submitting;
  if (!submission.phase.compare_exchange_strong(
          expected, ResidencySubmissionPhase::Completed,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  const ResidencyPipelineCompletion completion = submission.completion;
  void *const user = submission.user;
  Status terminal = failure;
  const std::shared_ptr<PipelineState> state = submission.pipeline;
  if (state != nullptr) {
    std::lock_guard lock{state->gate};
    if (state->phase == PipelinePhase::Running) {
      PipelineOutcome rejected{.status = failure,
                               .publication_suppressed = true};
      rejected.defer_generation = submission.defer_generation;
      terminal = submission.defer_generation
                     ? publish_residency_pipeline_outcome(
                           *state, submission.issued_steps, rejected, true,
                           submission.publication_generation,
                           submission.publication_parity)
                     : publish_residency_pipeline_execution(
                           *state, submission.issued_steps, rejected);
    } else if (failure) {
      terminal = Status::fail(Reason::CompletionInvalid);
    }
  }
  submission.completion = nullptr;
  submission.user = nullptr;
  if (completion != nullptr && user != nullptr) {
    completion(user, terminal);
  }
}

void complete_residency_pipeline_submission(
    void *const raw,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  auto *const submission = static_cast<ResidencyPipelineSubmission *>(raw);
  if (submission == nullptr) {
    return;
  }
  ResidencySubmissionPhase expected = ResidencySubmissionPhase::Submitting;
  if (!submission->phase.compare_exchange_strong(
          expected, ResidencySubmissionPhase::Completed,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    expected = ResidencySubmissionPhase::Submitted;
    if (!submission->phase.compare_exchange_strong(
            expected, ResidencySubmissionPhase::Completed,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }
  if (submission->completion == nullptr || submission->user == nullptr ||
      submission->issued_steps == 0u ||
      submission->issued_steps > submission->locals.size()) {
    return;
  }
  const std::shared_ptr<PipelineState> state = submission->pipeline;
  const std::span<const std::uint32_t> locals{submission->locals.data(),
                                              submission->issued_steps};
  const ResidencyPipelineCompletion completion = submission->completion;
  void *const user = submission->user;
  Status status = Status::fail(Reason::PipelineInvalid);
  {
    std::lock_guard lock{state->gate};
    if (state->phase == PipelinePhase::Running) {
      PipelineOutcome outcome{.writes_possible = submission->writes_possible,
                              .publication_suppressed = true};
      outcome = finish_residency_pipeline_execution(*state, locals, evidence,
                                                    outcome);
      outcome.defer_generation = submission->defer_generation;
      status = submission->defer_generation
                   ? publish_residency_pipeline_outcome(
                         *state, submission->issued_steps, outcome, true,
                         submission->publication_generation,
                         submission->publication_parity)
                   : publish_residency_pipeline_execution(
                         *state, submission->issued_steps, outcome);
    }
  }
  submission->completion = nullptr;
  submission->user = nullptr;
  completion(user, status);
}

} // namespace rund::compute::detail
