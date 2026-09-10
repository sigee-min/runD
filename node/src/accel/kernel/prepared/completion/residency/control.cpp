#include "internal.hpp"

#include <algorithm>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

rund::AccelCheck SignalPreparedKernelPipelineWindow(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
  auto *const state =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || state == nullptr ||
      !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
      state->ops->signal_prepared_window == nullptr ||
      state->backend == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (signal.control_generation == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  PreparedResidencyWindowControl *control = nullptr;
  {
    std::lock_guard submission_lock{state->submission.mutex};
    control =
        static_cast<PreparedResidencyWindowControl *>(state->submission.window);
  }
  if (control == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  {
    std::lock_guard control_lock{control->gate};
    if (!control->active || control->quarantined || control->aborting ||
        signal.plan_identity != control->bound.plan_identity ||
        signal.token != control->bound.token ||
        signal.generation != control->bound.generation ||
        signal.epoch < control->bound.first_epoch ||
        signal.epoch - control->bound.first_epoch >=
            control->bound.batch_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const PreparedResidencyWindowBatch &batch = control->bound.batches[
        static_cast<std::size_t>(signal.epoch - control->bound.first_epoch)];
    if (batch.pipeline.owner != prepared.owner || batch.bank != signal.bank ||
        batch.control_generation != signal.control_generation) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  // Native readiness validation owns the seed transition. Seeding here would
  // mutate an executing Pipeline before a backend rejects a duplicate, stale,
  // or too-early recurrent-bank signal. Metal and Vulkan also use different
  // native recurrence strides, so this common layer authenticates immutable
  // credentials only and delegates the atomic validate+seed+ready operation.
  // Never retain or touch caller-owned `control` after invoking the backend.
  // An admitted backend may complete Release/Final inline; those callbacks
  // must be able to acquire control.gate and may let the caller destroy the
  // whole control immediately after their terminal notification.
  return state->ops->signal_prepared_window(state->backend, signal);
}

rund::AccelCheck AbortPreparedKernelPipelineWindow(
    const rund::AccelContext &context,
    PreparedResidencyWindowControl &control,
    const BackendResidencyWindowAbort &abort) noexcept {
  if (abort.failure.ok || abort.failure.reason == nullptr ||
      abort.failure.reason[0u] == '\0') {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const BackendOps *ops = nullptr;
  std::shared_ptr<void> leader{};
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.quarantined || control.aborting ||
        control.bound.batch_count == 0u ||
        abort.plan_identity != control.bound.plan_identity ||
        abort.token != control.bound.token ||
        abort.generation != control.bound.generation) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const PreparedKernelPipeline &prepared = control.bound.batches[0u].pipeline;
    auto *const state =
        static_cast<prepared::PipelineState *>(prepared.owner.get());
    const bool claimed =
        std::find(control.states.begin(), control.states.end(), state) !=
        control.states.end();
    if (!prepared.ok || state == nullptr ||
        !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
        state->ops->abort_prepared_window == nullptr ||
        state->backend == nullptr || !claimed) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    ops = state->ops;
    leader = state->backend;
    control.aborting = true;
  }
  const rund::AccelCheck drained = ops->abort_prepared_window(leader, abort);
  if (!drained.ok) {
    std::lock_guard lock{control.gate};
    if (control.active) {
      control.aborting = false;
    }
  }
  return drained;
}

} // namespace rund::node::accel::detail
