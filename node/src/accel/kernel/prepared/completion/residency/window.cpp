#include "internal.hpp"

#include <array>
#include <mutex>

namespace rund::node::accel::detail {

rund::AccelCheck SubmitPreparedKernelPipelineWindow(
    const rund::AccelContext &context,
    const PreparedResidencyWindowRequest &request,
    PreparedResidencyWindowControl &control) noexcept {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  const BackendOps *ops = nullptr;
  const std::size_t state_count = prepared_residency::WindowStates(
      request, states, context, ops);
  if (state_count == 0u || ops == nullptr) {
    return invalid;
  }
  std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> claims{};
  for (std::size_t index = 0u; index < state_count; ++index) {
    claims[index] = std::unique_lock<std::mutex>{
        states[index]->submission.mutex, std::try_to_lock};
    if (!claims[index].owns_lock()) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
  }
  std::unique_lock control_lock{control.gate, std::try_to_lock};
  if (!control_lock.owns_lock() || control.active) {
    return rund::AccelCheck{false, "compute_pipeline_busy"};
  }
  for (std::size_t index = 0u; index < state_count; ++index) {
    const prepared::PipelineSubmission &submission =
        states[index]->submission;
    const bool stream_claim =
        control.stream != nullptr && submission.stream == control.stream &&
        submission.owner != nullptr && !submission.quarantined &&
        submission.window == nullptr;
    // This revalidation runs under every prepared submission mutex. It closes
    // the StreamWindow validate->raw-submit gap: a concurrent stream release
    // can remove the sentinel before this point, but can never launch an
    // ownerless native window afterward.
    if ((control.stream != nullptr && !stream_claim) ||
        (submission.active() && !stream_claim)) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
  }
  control.bound = request;
  control.backend = BackendResidencyWindowRequest{
      .plan_identity = request.plan_identity,
      .token = request.token,
      .generation = request.generation,
      .first_epoch = request.first_epoch,
      .batch_count = request.batch_count,
      .release = prepared_residency::CompleteResidencyWindowRelease,
      .final = prepared_residency::CompleteResidencyWindowFinal,
      .user = &control,
  };
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    const PreparedResidencyWindowBatch &source = request.batches[index];
    auto *const state =
        static_cast<prepared::PipelineState *>(source.pipeline.owner.get());
    control.backend.batches[index] = BackendResidencyWindowBatch{
        .prepared = state->backend,
        .locals = source.locals,
        .local_count = source.local_count,
        .epoch = source.epoch,
        .control_generation = source.control_generation,
        .bank = source.bank,
    };
  }
  control.states = states;
  control.receipts = {};
  control.release_count = 0u;
  control.aborting = false;
  control.quarantined = false;
  control.active = true;
  for (std::size_t index = 0u; index < state_count; ++index) {
    // The submission is the strong quarantine owner. The caller-owned
    // PreparedResidencyWindowControl may die after an Unknown Final while a
    // native command can still view this prepared Pipeline.
    if (control.stream == nullptr) {
      for (std::size_t batch = 0u; batch < request.batch_count; ++batch) {
        if (request.batches[batch].pipeline.owner.get() == states[index]) {
          states[index]->submission.owner =
              request.batches[batch].pipeline.owner;
          break;
        }
      }
    }
    states[index]->submission.window = &control;
  }
  control_lock.unlock();
  for (std::size_t index = 0u; index < state_count; ++index) {
    claims[index].unlock();
  }

  const rund::AccelCheck submitted =
      ops->submit_prepared_window(control.backend);
  if (submitted.ok) {
    return submitted;
  }

  // Match the normal claim order: submission mutexes, then control.gate.
  std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> rollback{};
  for (std::size_t index = 0u; index < state_count; ++index) {
    rollback[index] =
        std::unique_lock<std::mutex>{states[index]->submission.mutex};
  }
  std::lock_guard control_guard{control.gate};
  for (std::size_t index = 0u; index < state_count; ++index) {
    if (states[index]->submission.window == &control) {
      states[index]->submission.window = nullptr;
    }
    if (control.stream == nullptr) {
      states[index]->submission.owner.reset();
      states[index]->submission.quarantined = false;
    }
  }
  control.bound = {};
  control.backend = {};
  control.states.fill(nullptr);
  control.receipts = {};
  control.release_count = 0u;
  control.stream = nullptr;
  control.active = false;
  control.aborting = false;
  control.quarantined = false;
  return submitted;
}

} // namespace rund::node::accel::detail
