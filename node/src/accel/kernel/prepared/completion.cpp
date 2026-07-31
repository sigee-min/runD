#include "../prepared.hpp"

#include "evidence.hpp"
#include "model.hpp"

#include "../evidence.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {
namespace {

void CompletePipeline(void *const raw, const KernelResult backend) noexcept {
  auto *const submission = static_cast<prepared::PipelineSubmission *>(raw);
  if (submission == nullptr) {
    return;
  }
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> lifetime{};
  PreparedPipelineCompletion completion = nullptr;
  void *user = nullptr;
  PreparedPipelineEvidence result{};
  {
    std::lock_guard lock{submission->mutex};
    prepared::PipelineState *const state = submission->pipeline();
    if (state == nullptr || submission->completion == nullptr) {
      return;
    }
    prepared::PipelineState &pipeline = *state;
    owner = std::move(submission->owner);
    lifetime = std::move(submission->lifetime);
    completion = submission->completion;
    user = submission->user;
    result = prepared::PipelineEvidence(pipeline.context, pipeline, backend);
    submission->completion = nullptr;
    submission->user = nullptr;
  }
  static_cast<void>(owner);
  static_cast<void>(lifetime);
  completion(user, std::move(result));
}

void CompleteRun(void *const raw, const KernelResult run) noexcept {
  auto *const submission = static_cast<prepared::RunSubmission *>(raw);
  if (submission == nullptr) {
    return;
  }
  rund::AccelContext context{};
  prepared::RunState *state = nullptr;
  PreparedKernelCompletion completion = nullptr;
  void *user = nullptr;
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> lifetime{};
  {
    std::lock_guard lock{submission->mutex};
    if (!submission->active || submission->prepared == nullptr ||
        submission->completion == nullptr) {
      return;
    }
    context = submission->context;
    state = submission->prepared;
    completion = submission->completion;
    user = submission->user;
    owner = std::move(submission->owner);
    lifetime = std::move(submission->lifetime);
    submission->context = {};
    submission->prepared = nullptr;
    submission->completion = nullptr;
    submission->user = nullptr;
  }
  const rund::AccelEvidence evidence =
      prepared::RunEvidence(context, *state, run);
  {
    std::lock_guard lock{submission->mutex};
    submission->active = false;
  }
  completion(user, evidence);
  static_cast<void>(owner);
  static_cast<void>(lifetime);
}

struct RunWait final {
  std::atomic_bool done{false};
  rund::AccelEvidence evidence{};
};

void CompleteRunWait(void *const raw,
                     const rund::AccelEvidence &evidence) noexcept {
  auto *const wait = static_cast<RunWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->evidence = evidence;
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

struct PipelineWait final {
  std::atomic_bool done{false};
  PreparedPipelineEvidence evidence{};
};

void CompletePipelineWait(void *const raw,
                          PreparedPipelineEvidence &&evidence) noexcept {
  auto *const wait = static_cast<PipelineWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->evidence = std::move(evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

} // namespace

rund::AccelCheck SubmitPreparedKernel(const rund::AccelContext &context,
                                      const PreparedKernelRun &prepared,
                                      std::shared_ptr<void> lifetime,
                                      const PreparedKernelCompletion completion,
                                      void *const user) noexcept {
  auto *const state = static_cast<prepared::RunState *>(prepared.owner.get());
  if (!prepared.ok || state == nullptr || completion == nullptr ||
      IsPipelinePrivatePreparation(state->mode) ||
      state->bound.run.ops == nullptr ||
      state->bound.run.ops->submit_prepared == nullptr ||
      !prepared::MatchesContext(context, *state)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  prepared::RunSubmission &submission = state->submission;
  {
    std::lock_guard lock{submission.mutex};
    if (submission.active) {
      return rund::AccelCheck{false, "compute_job_busy"};
    }
    submission.context = context;
    submission.prepared = state;
    submission.owner = prepared.owner;
    submission.lifetime = std::move(lifetime);
    submission.completion = completion;
    submission.user = user;
    submission.active = true;
  }
  const rund::AccelCheck submitted = state->bound.run.ops->submit_prepared(
      state->bound.run, state->backend, CompleteRun, &submission,
      &state->memory, prepared.owner);
  if (!submitted.ok) {
    std::lock_guard lock{submission.mutex};
    if (submission.active) {
      submission.context = {};
      submission.prepared = nullptr;
      submission.owner.reset();
      submission.lifetime.reset();
      submission.completion = nullptr;
      submission.user = nullptr;
      submission.active = false;
    }
  }
  return submitted;
}

rund::AccelCheck SubmitPreparedKernelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> lifetime, const PreparedPipelineCompletion completion,
    void *const user) noexcept {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr ||
      !prepared::ValidPipeline(context, *pipeline) || completion == nullptr ||
      user == nullptr || pipeline->ops->submit_prepared_pipeline == nullptr) {
    return invalid;
  }
  prepared::PipelineSubmission &submission = pipeline->submission;
  {
    std::lock_guard lock{submission.mutex};
    if (submission.active()) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
    // Private prepared steps are owned by this Pipeline. Its state gate and
    // this owning submission claim are the complete concurrency authority.
    submission.owner = prepared.owner;
    submission.lifetime = std::move(lifetime);
    submission.completion = completion;
    submission.user = user;
  }
  const rund::AccelCheck submitted = pipeline->ops->submit_prepared_pipeline(
      pipeline->backend, CompletePipeline, &submission);
  if (!submitted.ok) {
    std::lock_guard lock{submission.mutex};
    if (submission.active()) {
      submission.owner.reset();
      submission.lifetime.reset();
      submission.completion = nullptr;
      submission.user = nullptr;
    }
  }
  return submitted;
}

rund::AccelEvidence RunPreparedKernel(const rund::AccelContext &context,
                                      const PreparedKernelRun &prepared) {
  const auto *const state =
      static_cast<const prepared::RunState *>(prepared.owner.get());
  RunWait wait{};
  const rund::AccelCheck submitted =
      SubmitPreparedKernel(context, prepared, {}, CompleteRunWait, &wait);
  if (!submitted.ok) {
    return RejectKernelEvidence(
        context, state == nullptr ? KernelExecution{} : state->execution,
        submitted.reason);
  }
  wait.done.wait(false, std::memory_order_acquire);
  return wait.evidence;
}

PreparedPipelineEvidence
RunPreparedKernelPipeline(const rund::AccelContext &context,
                          const PreparedKernelPipeline &prepared) {
  PipelineWait wait{};
  const rund::AccelCheck submitted = SubmitPreparedKernelPipeline(
      context, prepared, {}, CompletePipelineWait, &wait);
  if (!submitted.ok) {
    return PreparedPipelineEvidence{.check = submitted};
  }
  wait.done.wait(false, std::memory_order_acquire);
  return std::move(wait.evidence);
}

} // namespace rund::node::accel::detail
