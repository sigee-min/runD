#include "internal.hpp"

#include "../../../compute/virtual/run/execution.hpp"
#include "../../../compute/virtual/state.hpp"
#include "../../runtime/local.hpp"

#include <utility>

namespace rund::node::compute_operation_detail {

using VirtualState = compute::detail::VirtualPipelineState;

namespace {

[[nodiscard]] std::shared_ptr<VirtualState>
VirtualOwner(const std::shared_ptr<void> &owner) noexcept {
  return std::static_pointer_cast<VirtualState>(owner);
}

} // namespace

[[nodiscard]] compute::Result<compute::Backend>
VirtualBackend(const std::shared_ptr<void> &owner) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(owner);
  return state == nullptr || state->pipeline == nullptr
             ? compute::Result<compute::Backend>::fail(
                   compute::Reason::PipelineInvalid)
             : compute::detail::pipeline_backend(state->pipeline);
}

[[nodiscard]] kernel::u32
VirtualWorkers(const std::shared_ptr<void> &owner) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(owner);
  return state == nullptr || state->pipeline == nullptr
             ? 0u
             : compute::detail::pipeline_workers(state->pipeline);
}

[[nodiscard]] compute::Status
ReserveVirtual(const std::shared_ptr<void> &owner) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(owner);
  return compute::detail::valid_virtual_pipeline(state)
             ? compute::Status::success()
             : compute::Status::fail(compute::Reason::PipelineInvalid);
}

[[nodiscard]] compute::Status
RunVirtual(const compute_detail::Operation &operation,
           compute_detail::TaskState &task) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(operation.owner);
  if (state == nullptr ||
      !compute_detail::ClaimVirtualStart(task.terminal_phase)) {
    return task.terminal_phase.load(std::memory_order_acquire) ==
                   compute_detail::TerminalPhase::Cancelled
               ? compute::Status::fail(compute::Reason::Cancelled)
               : compute::Status::fail(compute::Reason::AlreadyCompleted);
  }
  if (state->pipeline != nullptr && state->pipeline->device != nullptr &&
      state->pipeline->device->backend != compute::Backend::Cpu) {
    const compute::detail::VirtualAsyncStart started =
        compute::detail::start_virtual_pipeline_async(state, &task, CpuReady);
    if (started.frame != nullptr) {
      task.virtual_continuation = std::move(started.frame);
      if (!started.status) {
        compute::Stats stats{};
        const compute::Status result =
            compute::detail::resume_virtual_pipeline_async(
                task.virtual_continuation, stats);
        {
          std::lock_guard lock{task.mutex};
          task.status = result;
          task.stats = stats;
        }
        task.completion_phase.store(3u, std::memory_order_release);
        return compute::Status::success();
      }
      return compute::Status::success();
    }
    if (started.status.reason() != compute::Reason::BackendUnsupported) {
      return started.status;
    }
  }
  const compute::Status result = compute::detail::run_virtual_pipeline(state);
  const compute::Stats stats = compute::detail::virtual_pipeline_stats(state);
  {
    std::lock_guard lock{task.mutex};
    task.status = result;
    task.stats = stats;
  }
  task.completion_phase.store(3u, std::memory_order_release);
  return result;
}

[[nodiscard]] compute_detail::Dispatch
SubmitVirtual(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept {
  const compute::Status result = RunVirtual(operation, task);
  if (!result) {
    return compute_detail::Dispatch::failed(result);
  }
  if (task.virtual_continuation != nullptr) {
    return compute_detail::Dispatch::backend_submitted();
  }
  std::lock_guard lock{task.mutex};
  return task.stats.command_submits == 0u
             ? compute_detail::Dispatch::accepted_without_backend()
             : compute_detail::Dispatch::backend_submitted();
}

[[nodiscard]] compute_detail::Advance
AdvanceVirtual(const compute_detail::Operation &,
               compute_detail::TaskState &task) noexcept {
  if (task.virtual_continuation != nullptr) {
    compute::Stats stats{};
    const compute::Status result =
        compute::detail::resume_virtual_pipeline_async(
            task.virtual_continuation, stats);
    {
      std::lock_guard lock{task.mutex};
      task.status = result;
      task.stats = stats;
    }
    return result ? compute_detail::Advance::complete()
                  : compute_detail::Advance::failed(result);
  }
  return compute_detail::Advance::complete();
}

[[nodiscard]] compute::detail::TerminalObservation
ResultVirtual(const compute_detail::Operation &,
              compute_detail::TaskState &task) noexcept {
  std::lock_guard lock{task.mutex};
  return compute::detail::TerminalObservationAccess::from_stats(task.status,
                                                                task.stats);
}

[[nodiscard]] compute::detail::TerminalObservation
FailVirtual(const compute_detail::Operation &operation,
            compute_detail::TaskState &, const compute::Status failure) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(operation.owner);
  return compute::detail::TerminalObservationAccess::from_stats(
      failure, compute::detail::virtual_pipeline_stats(state));
}

[[nodiscard]] compute::detail::TerminalObservation
CancelVirtual(const compute_detail::Operation &operation,
              compute_detail::TaskState &) noexcept {
  const std::shared_ptr<VirtualState> state = VirtualOwner(operation.owner);
  return compute::detail::TerminalObservationAccess::from_stats(
      compute::Status::fail(compute::Reason::Cancelled),
      compute::detail::virtual_pipeline_stats(state));
}

} // namespace rund::node::compute_operation_detail
