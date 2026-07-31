#include "../compute/state.hpp"
#include "../replay/scope/plan.hpp"
#include "local.hpp"

namespace rund::node {

using runtime_detail::Runnable;
using runtime_detail::RuntimeActive;

namespace runtime_detail {

rund::kernel::ParallelRuntime
AcquireParallelRuntime(void *const context, const rund::kernel::u32 workers) {
  auto *const runtime = static_cast<Runtime *>(context);
  if (runtime == nullptr) {
    return rund::kernel::ParallelRuntime{
        .workers = workers,
        .reason = "parallel_runtime_missing",
    };
  }
  auto &state = RuntimeAccess::state(*runtime);
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.lifecycle == ::rund::SessionState::Unconfigured) {
    return rund::kernel::ParallelRuntime{
        .workers = workers,
        .reason = "not_configured",
    };
  }
  if (!Runnable(state.lifecycle)) {
    return rund::kernel::ParallelRuntime{
        .workers = workers,
        .reason = "not_runnable",
    };
  }
  const ResourceEnvelope &resources = state.resources;
  const rund::kernel::u32 requested_workers =
      workers == 0u ? resources.observed.workers : workers;
  if (resources.observed.workers != requested_workers) {
    return rund::kernel::ParallelRuntime{
        .workers = requested_workers,
        .reason = "parallel_runtime_width_mismatch",
    };
  }
  if (!static_cast<bool>(resources.worker_backend)) {
    return rund::kernel::ParallelRuntime{
        .workers = workers,
        .reason = "backend_invalid",
    };
  }
  thread_local rund::kernel::Workspace workspace{};
  return rund::kernel::ParallelRuntime{
      .workspace = &workspace,
      .worker_backend = resources.worker_backend,
      .workers = resources.observed.workers,
      .valid = true,
      .reason = "ok",
  };
}

rund::kernel::executor_detail::ScopedParallelRuntimeProvider
InstallKernelScope(Runtime &runtime) {
  return rund::kernel::executor_detail::ScopedParallelRuntimeProvider(
      rund::kernel::ParallelRuntimeProvider{
          .context = &runtime,
          .acquire = AcquireParallelRuntime,
      });
}

} // namespace runtime_detail

Runtime::TaskScopeFrame
Runtime::task_scope(
    const std::shared_ptr<runtime_detail::ComputeHostState> &host,
    const ::rund::replay::detail::scope::Plan &plan) {
  return TaskScopeFrame{std::static_pointer_cast<void>(host), &host->scheduler,
                        host->control, plan};
}

} // namespace rund::node
