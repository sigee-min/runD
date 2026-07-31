#include "state.hpp"

namespace rund::kernel {

ComputeTileRunResult
ComputeTileExecutor::run_erased(const void *const callback_context,
                                const ComputeTileCallback callback) {
  return run_with_erased(state_ == nullptr ? WorkerBackend{} : state_->backend,
                         callback_context, callback);
}

ComputeTileRunResult
ComputeTileExecutor::run_with_erased(const WorkerBackend worker_backend,
                                     const void *const callback_context,
                                     const ComputeTileCallback callback) {
  if (!prepared()) {
    return ComputeTileRunResult{
        .reason = state_ == nullptr ? "compute_tile_executor_capacity"
                                    : state_->reason,
    };
  }
  if (state_->prepared.units == 0u) {
    return ComputeTileRunResult{
        .ok = true,
        .reason = "pass",
        .worker_count = state_->workers,
    };
  }
  const compute_tile_detail::Backend selected = compute_tile_detail::Select(
      worker_backend, state_->workers, compute_tile_detail::Mode::Sync);
  if (!selected.ok) {
    return ComputeTileRunResult{.reason = selected.reason};
  }

  compute_tile_detail::Context context{};
  compute_tile_detail::Begin(context, callback_context, callback,
                             state_->prepared, state_->failures,
                             state_->worker_tiles);
  const RunResult run = RunPreparedProgram(RunPreparedProgramRequest{
      .workspace = &state_->workspace,
      .worker_backend = selected.value,
      .context = &context,
      .dispatch = compute_tile_detail::Invoke,
      .collect_worker_stats = true,
      .minimum_partition_count = 1u,
      .require_no_allocation = true,
  });
  const Telemetry &stats = run.kernel.telemetry;
  const u32 units = state_->prepared.physical_tiling_enabled
                        ? static_cast<u32>(state_->prepared.physical_tile_units)
                        : count();
  return compute_tile_detail::Project(
      context,
      compute_tile_detail::Completion{
          .ok = run.ok,
          .reason = run.reason,
          .worker_count = stats.worker_count,
          .dispatch_count = stats.backend_dispatch_count,
      },
      count(), tile_count(), units);
}

} // namespace rund::kernel
