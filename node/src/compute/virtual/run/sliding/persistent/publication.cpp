#include "../../../../../compute/device/residency/execution/sliding/final/abort.hpp"
#include "../internal.hpp"

#include "../../../../pipeline/claim.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

std::atomic_bool preflight_failure_once{};
std::atomic<void (*)() noexcept> publication_pause_once{};

[[nodiscard]] bool
valid_backing_publication(const SlidingProductRun &state) noexcept {
  return state.output != nullptr && state.run != nullptr && state.recovery &&
         state.run->active.output_bytes != 0u &&
         state.run->output_version ==
             VirtualBackingAccess::version(*state.output) &&
         VirtualBackingAccess::recovery_bytes(*state.output) ==
             state.run->active.output_bytes;
}

[[nodiscard]] bool valid_pipeline_publication(const SlidingProductRun &state,
                                              const std::size_t bank) noexcept {
  const PipelineState &pipeline = *state.cold->pipelines[bank];
  const PipelinePublicationState &publication = *pipeline.publication;
  return state.pipeline_started[bank] && state.pipeline_submitted[bank] &&
         pipeline.phase == PipelinePhase::Running &&
         !pipeline.control_poisoned && !publication.device_lost &&
         publication.attempt_active &&
         publication.generation == pipeline.attempt.generation &&
         publication.parity == pipeline.attempt.parity &&
         pipeline.native_generation == publication.generation + 1u &&
         pipeline.native_parity == publication.parity &&
         has_private_residency_authority(pipeline);
}

[[nodiscard]] bool
lock_and_validate_pipelines(SlidingProductRun &state,
                            PersistentPublication &publication) noexcept {
  if (state.cold == nullptr || state.cold->pipelines[0u] == nullptr ||
      state.cold->pipelines[1u] == nullptr ||
      state.cold->pipelines[0u] == state.cold->pipelines[1u] ||
      state.cold->pipelines[0u]->publication == nullptr ||
      state.cold->pipelines[1u]->publication == nullptr ||
      state.cold->pipelines[0u]->publication ==
          state.cold->pipelines[1u]->publication) {
    return false;
  }
  PipelineState *const bank0 = state.cold->pipelines[0u].get();
  PipelineState *const bank1 = state.cold->pipelines[1u].get();
  PipelineState *const first =
      std::less<PipelineState *>{}(bank0, bank1) ? bank0 : bank1;
  PipelineState *const second = first == bank0 ? bank1 : bank0;
  publication.first_pipeline = std::unique_lock<std::mutex>{first->gate};
  publication.second_pipeline = std::unique_lock<std::mutex>{second->gate};
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    std::lock_guard publication_lock{
        state.cold->pipelines[bank]->publication->gate};
    if (!valid_pipeline_publication(state, bank)) {
      return false;
    }
  }
  return true;
}

void unlock_pipelines(PersistentPublication &publication) noexcept {
  if (publication.second_pipeline.owns_lock()) {
    publication.second_pipeline.unlock();
  }
  if (publication.first_pipeline.owns_lock()) {
    publication.first_pipeline.unlock();
  }
}

void publish_persistent_backing(void *const raw) noexcept {
  auto &publication = *static_cast<PersistentPublication *>(raw);
  SlidingProductRun &state = *publication.state;
  if (const auto pause =
          publication_pause_once.exchange(nullptr, std::memory_order_acq_rel)) {
    pause();
  }
  publish_residency_cache(*state.output);
  if (VirtualBackingAccess::transaction_provider(*state.output) == nullptr) {
    VirtualBackingAccess::clear_recovery(*state.output);
  }
  fold_service_stats(state, publication.evidence);
}

} // namespace

Status prepare_persistent_publication(
    SlidingProductRun &state,
    const node::accel::detail::PersistentResidencySlidingFinal &native,
    PersistentPublication &publication) noexcept {
  if (!valid_backing_publication(state) || state.pool == nullptr) {
    return Status::fail(Reason::CompletionInvalid);
  }
  if (preflight_failure_once.exchange(false, std::memory_order_acq_rel)) {
    publication.close = CloseRequirement::Required;
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::execution::SlidingFinal frozen{};
  residency::ExecutionSlidingFinal authority_final{};
  residency::FinalAbort abort = residency::FinalAbort::Invalid;
  auto sliding = state.pool->authority().sliding();
  if (!state.sliding.prepare_final(frozen)) {
    publication.abort = sliding.close_execution_sliding(
        state.cold->plan, state.sliding,
        Status::fail(Reason::CompletionInvalid));
    return Status::fail(Reason::CompletionInvalid);
  }
  if (!sliding.prepare_execution_sliding(
          state.cold->plan, state.sliding, frozen, authority_final, &abort)) {
    publication.abort = abort;
    return Status::fail(Reason::CompletionInvalid);
  }
  const RebaseResult rebased =
      rebase_pipeline_sliding(state, Status::success());
  if (rebased.disposition == RebaseDisposition::Poisoned) {
    publication.abort = sliding.abort_execution_sliding(
        state.cold->plan, state.sliding, frozen, &authority_final,
        Status::fail(Reason::PipelinePoisoned));
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (rebased.disposition == RebaseDisposition::RestoredFailure) {
    publication.abort = sliding.abort_execution_sliding(
        state.cold->plan, state.sliding, frozen, &authority_final,
        rebased.status);
    return rebased.status;
  }
  if (!lock_and_validate_pipelines(state, publication)) {
    unlock_pipelines(publication);
    const RebaseResult restored =
        rebase_pipeline_sliding(state, Status::fail(Reason::CompletionInvalid));
    const Status failure = restored.disposition == RebaseDisposition::Poisoned
                               ? Status::fail(Reason::PipelinePoisoned)
                               : restored.status;
    publication.abort = sliding.abort_execution_sliding(
        state.cold->plan, state.sliding, frozen, &authority_final, failure);
    return failure;
  }

  publication.state = &state;
  publication.native = &native;
  const residency::ExecutionClose closed =
      sliding.accept_execution_sliding_final(
          state.cold->plan, state.sliding, frozen, std::move(authority_final),
          publication.evidence, &publication, commit_persistent_publication);
  if (!closed || !closed.success || !publication.evidence.status) {
    unlock_pipelines(publication);
    publication.abort = closed.quarantined ? residency::FinalAbort::Quarantined
                        : closed           ? residency::FinalAbort::Closed
                                           : residency::FinalAbort::Invalid;
    return Status::fail(Reason::CompletionInvalid);
  }
  return Status::success();
}

void commit_persistent_publication(void *const raw) noexcept {
  auto &publication = *static_cast<PersistentPublication *>(raw);
  SlidingProductRun &state = *publication.state;
  // The accepted proof retains both Pipeline gates.  Every validation and
  // receipt operation has already completed; from the first terminal below
  // through backing version/recovery publication there is no failure return.
  const std::size_t issued = state.run->frame_capacity;
  PipelineState &first = *state.cold->pipelines[0u];
  PipelineState &second = *state.cold->pipelines[1u];
  publish_private_pipeline_terminal_pair(
      first, PipelineTerminal{.issued_steps = issued}, second,
      PipelineTerminal{.issued_steps = issued}, &publication,
      publish_persistent_backing);
}

void inject_persistent_publication_preflight_failure_once() noexcept {
  preflight_failure_once.store(true, std::memory_order_release);
}

void inject_persistent_publication_pause_once(
    void (*const pause)() noexcept) noexcept {
  publication_pause_once.store(pause, std::memory_order_release);
}

} // namespace rund::compute::detail::sliding_product_detail
