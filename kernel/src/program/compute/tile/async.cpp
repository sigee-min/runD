#include "state.hpp"

namespace rund::kernel {

ComputeTileSubmitResult ComputeTileExecutor::submit_with_erased(
    const WorkerBackend worker_backend, const void *const callback_context,
    const ComputeTileCallback callback, void *const ready_context,
    const ComputeTileReady ready) noexcept {
  if (plan_ == nullptr || !plan_->prepared.valid) {
    return ComputeTileSubmitResult{.reason = reason_};
  }
  if (storage_ == nullptr) {
    return ComputeTileSubmitResult{
        .reason = "compute_tile_run_storage_missing",
    };
  }
  if (storage_->generation_.load(std::memory_order_acquire) !=
      bound_generation_) {
    return ComputeTileSubmitResult{.reason = "compute_tile_run_rebound"};
  }
  if (callback == nullptr || ready == nullptr) {
    return ComputeTileSubmitResult{.reason = "compute_tile_callback_invalid"};
  }
  const compute_tile_detail::Backend selected = compute_tile_detail::Select(
      worker_backend, plan_->workers, compute_tile_detail::Mode::Async);
  if (!selected.ok) {
    return ComputeTileSubmitResult{.reason = selected.reason};
  }

  std::uint8_t expected =
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle);
  if (!storage_->phase_.compare_exchange_strong(
          expected,
          static_cast<std::uint8_t>(compute_tile_detail::Phase::Async),
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return ComputeTileSubmitResult{.reason = "compute_tile_run_busy"};
  }
  if (storage_->generation_.load(std::memory_order_acquire) !=
      bound_generation_) {
    storage_->phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    return ComputeTileSubmitResult{.reason = "compute_tile_run_rebound"};
  }

  compute_tile_detail::Begin(*storage_, callback_context, callback);
  storage_->ready_context_ = ready_context;
  storage_->ready_ = ready;
  storage_->async_ok_ = false;
  storage_->submission_.remaining.store(0u, std::memory_order_relaxed);
  storage_->submission_.failed.store(false, std::memory_order_relaxed);
  storage_->submission_.completion = WorkerCompletion{
      .context = storage_,
      .invoke = CompleteSubmission,
  };
  if (storage_->active_count_ == 0u) {
    CompleteSubmission(storage_, true);
    return ComputeTileSubmitResult{.ok = true, .reason = "pass"};
  }
  const ScheduleView schedule = storage_->workspace_.program.schedule;
  const bool submitted = selected.value.submit_partitions(
      selected.value.context, schedule.partitions, schedule.partition_count,
      WorkerTask{.context = storage_,
                 .invoke = compute_tile_detail::InvokeWorker},
      nullptr, &storage_->submission_);
  if (!submitted) {
    storage_->ready_ = nullptr;
    storage_->ready_context_ = nullptr;
    storage_->phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    return ComputeTileSubmitResult{.reason = "compute_tile_submit_failed"};
  }
  return ComputeTileSubmitResult{.ok = true, .reason = "pass"};
}

void ComputeTileExecutor::CompleteSubmission(void *const raw,
                                             const bool ok) noexcept {
  auto *const storage = static_cast<ComputeTileRunStorage *>(raw);
  if (storage == nullptr) {
    return;
  }
  std::uint8_t expected =
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Async);
  if (!storage->phase_.compare_exchange_strong(
          expected,
          static_cast<std::uint8_t>(compute_tile_detail::Phase::Binding),
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  storage->async_ok_ = ok;
  storage->phase_.store(
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Ready),
      std::memory_order_release);
  if (storage->ready_ != nullptr) {
    storage->ready_(storage->ready_context_);
  }
}

ComputeTileRunResult ComputeTileExecutor::finish() noexcept {
  if (plan_ == nullptr || storage_ == nullptr) {
    return ComputeTileRunResult{.reason = "compute_tile_not_ready"};
  }
  if (storage_->generation_.load(std::memory_order_acquire) !=
      bound_generation_) {
    return ComputeTileRunResult{.reason = "compute_tile_run_rebound"};
  }
  std::uint8_t expected =
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Ready);
  if (!storage_->phase_.compare_exchange_strong(
          expected,
          static_cast<std::uint8_t>(compute_tile_detail::Phase::Binding),
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return ComputeTileRunResult{.reason = "compute_tile_not_ready"};
  }

  ComputeTileRunResult result = compute_tile_detail::Project(
      *storage_, storage_->async_ok_,
      storage_->async_ok_ ? "pass" : "compute_tile_backend_failed",
      plan_->workers, 1u, true);
  storage_->ready_ = nullptr;
  storage_->ready_context_ = nullptr;
  storage_->phase_.store(
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
      std::memory_order_release);
  return result;
}

} // namespace rund::kernel
