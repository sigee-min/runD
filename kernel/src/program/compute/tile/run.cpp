#include "state.hpp"

#include <span>

namespace rund::kernel {

ComputeTileRunResult
ComputeTileExecutor::run_erased(const void *const callback_context,
                                const ComputeTileCallback callback) {
  return run_with_erased(plan_ == nullptr ? WorkerBackend{} : plan_->backend,
                         callback_context, callback);
}

ComputeTileRunResult
ComputeTileExecutor::run_with_erased(const WorkerBackend worker_backend,
                                     const void *const callback_context,
                                     const ComputeTileCallback callback) {
  if (plan_ == nullptr || !plan_->prepared.valid) {
    return ComputeTileRunResult{.reason = reason_};
  }
  if (storage_ == nullptr) {
    return ComputeTileRunResult{.reason = "compute_tile_run_storage_missing"};
  }
  if (callback == nullptr) {
    return ComputeTileRunResult{.reason = "compute_tile_callback_invalid"};
  }
  if (storage_->generation_.load(std::memory_order_acquire) !=
      bound_generation_) {
    return ComputeTileRunResult{.reason = "compute_tile_run_rebound"};
  }

  const compute_tile_detail::Backend selected = compute_tile_detail::Select(
      worker_backend, plan_->workers, compute_tile_detail::Mode::Sync);
  if (!selected.ok) {
    return ComputeTileRunResult{.reason = selected.reason};
  }

  std::uint8_t expected =
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle);
  if (!storage_->phase_.compare_exchange_strong(
          expected, static_cast<std::uint8_t>(compute_tile_detail::Phase::Sync),
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return ComputeTileRunResult{.reason = "compute_tile_run_busy"};
  }
  if (storage_->generation_.load(std::memory_order_acquire) !=
      bound_generation_) {
    storage_->phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    return ComputeTileRunResult{.reason = "compute_tile_run_rebound"};
  }

  compute_tile_detail::Begin(*storage_, callback_context, callback);
  if (storage_->active_count_ == 0u) {
    storage_->phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    return ComputeTileRunResult{
        .ok = true,
        .reason = "pass",
        .worker_count = plan_->workers,
    };
  }

  RunResult run{};
  try {
    run = RunPreparedProgram(RunPreparedProgramRequest{
        .workspace = &storage_->workspace_,
        .worker_backend = selected.value,
        .context = storage_,
        .dispatch = compute_tile_detail::Invoke,
        .collect_worker_stats = true,
        .worker_stats_sink = std::span<u32>(storage_->worker_stats_partitions_,
                                            storage_->worker_count_),
        .worker_start_offset_ns_sink = std::span<u64>(
            storage_->worker_stats_start_offset_ns_, storage_->worker_count_),
        .worker_elapsed_ns_sink = std::span<u64>(
            storage_->worker_stats_elapsed_ns_, storage_->worker_count_),
        .worker_tail_wait_ns_sink = std::span<u64>(
            storage_->worker_stats_tail_wait_ns_, storage_->worker_count_),
        .minimum_partition_count = 1u,
        .require_no_allocation = true,
    });
  } catch (...) {
    storage_->phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    return ComputeTileRunResult{.reason = "compute_tile_backend_failed"};
  }
  const Telemetry &stats = run.kernel.telemetry;
  ComputeTileRunResult result = compute_tile_detail::Project(
      *storage_, run.ok, run.reason, stats.worker_count,
      stats.backend_dispatch_count, false);
  storage_->phase_.store(
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
      std::memory_order_release);
  return result;
}

} // namespace rund::kernel
