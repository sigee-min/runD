#include "state.hpp"

namespace rund::kernel {

ComputeTileSubmitResult ComputeTileExecutor::submit_with_erased(
    const WorkerBackend worker_backend, const void *const callback_context,
    const ComputeTileCallback callback, void *const ready_context,
    const ComputeTileReady ready) noexcept {
  if (!prepared() || state_->async_context == nullptr) {
    return ComputeTileSubmitResult{
        .reason = state_ == nullptr ? "compute_tile_executor_capacity"
                                    : state_->reason,
    };
  }
  if (callback == nullptr || ready == nullptr) {
    return ComputeTileSubmitResult{.reason = "compute_tile_callback_invalid"};
  }
  const compute_tile_detail::Backend selected = compute_tile_detail::Select(
      worker_backend, state_->workers, compute_tile_detail::Mode::Async);
  if (!selected.ok) {
    return ComputeTileSubmitResult{.reason = selected.reason};
  }
  std::uint8_t expected = 0u;
  if (!state_->async_phase.compare_exchange_strong(
          expected, 1u, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return ComputeTileSubmitResult{.reason = "compute_tile_run_busy"};
  }

  compute_tile_detail::Begin(*state_->async_context, callback_context, callback,
                             state_->prepared, state_->failures,
                             state_->worker_tiles);
  state_->ready_context = ready_context;
  state_->ready = ready;
  state_->async_ok = false;
  state_->submission.remaining.store(0u, std::memory_order_relaxed);
  state_->submission.failed.store(false, std::memory_order_relaxed);
  state_->submission.completion = WorkerCompletion{
      .context = state_.get(),
      .invoke = CompleteSubmission,
  };
  if (state_->prepared.units == 0u) {
    CompleteSubmission(state_.get(), true);
    return ComputeTileSubmitResult{.ok = true, .reason = "pass"};
  }
  const ScheduleView schedule = state_->workspace.program.schedule;
  const bool submitted = selected.value.submit_partitions(
      selected.value.context, schedule.partitions, schedule.partition_count,
      WorkerTask{.context = state_->async_context.get(),
                 .invoke = compute_tile_detail::InvokeWorker},
      nullptr, &state_->submission);
  if (!submitted) {
    state_->ready = nullptr;
    state_->ready_context = nullptr;
    state_->async_phase.store(0u, std::memory_order_release);
    return ComputeTileSubmitResult{.reason = "compute_tile_submit_failed"};
  }
  return ComputeTileSubmitResult{.ok = true, .reason = "pass"};
}

void ComputeTileExecutor::CompleteSubmission(void *const raw,
                                             const bool ok) noexcept {
  auto *const state = static_cast<State *>(raw);
  if (state == nullptr) {
    return;
  }
  std::uint8_t expected = 1u;
  if (!state->async_phase.compare_exchange_strong(
          expected, 2u, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  state->async_ok = ok;
  if (state->ready != nullptr) {
    state->ready(state->ready_context);
  }
}

ComputeTileRunResult ComputeTileExecutor::finish() noexcept {
  if (!prepared() || state_->async_context == nullptr ||
      state_->async_phase.load(std::memory_order_acquire) != 2u) {
    return ComputeTileRunResult{.reason = "compute_tile_not_ready"};
  }
  const u32 units = state_->prepared.physical_tiling_enabled
                        ? static_cast<u32>(state_->prepared.physical_tile_units)
                        : count();
  ComputeTileRunResult result = compute_tile_detail::Project(
      *state_->async_context,
      compute_tile_detail::Completion{
          .ok = state_->async_ok,
          .reason = state_->async_ok ? "pass" : "compute_tile_backend_failed",
          .worker_count = state_->workers,
          .dispatch_count = 1u,
          .report_tail_on_failure = true,
      },
      count(), tile_count(), units);
  state_->ready = nullptr;
  state_->ready_context = nullptr;
  state_->async_phase.store(0u, std::memory_order_release);
  return result;
}

} // namespace rund::kernel
