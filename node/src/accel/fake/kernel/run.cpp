#include "../../kernel/backend/execute.hpp"
#include "../../kernel/memory.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck RunFakeKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < run.step_count; ++index) {
    const BoundStep &bound = run.steps[index];
    if (bound.step == nullptr || bound.planned == nullptr ||
        bound.step->kind() != rund::kernel::NodeKind::Map ||
        bound.planned->artifact == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_backend_failed"};
    }
    const rund::kernel::BindingSet binding = MapBindingFor(bound);
    if (!ExecuteRetainedFake(*run.pick, bound.planned->plan,
                             *bound.planned->artifact, bound.map_windows.data(),
                             bound.map_windows.size(), binding)) {
      return rund::AccelCheck{false, "accel_kernel_backend_failed"};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck PrepareFakeKernel(const BackendRun &run,
                                   std::shared_ptr<void> &prepared,
                                   PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  return run.pick != nullptr && run.steps != nullptr && run.step_count != 0u
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

rund::AccelCheck
SubmitPreparedFakeKernel(const BackendRun &run, const std::shared_ptr<void> &,
                         const KernelCompletion completion, void *const user,
                         PreparedMemoryMeter *,
                         const std::shared_ptr<void> &) noexcept {
  if (completion == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  completion(user, KernelResult{
                       .check = RunFakeKernel(run),
                       .stats = {.dispatch_count = run.final_dispatch_count,
                                 .ok = true,
                                 .reason = "ok"},
                   });
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
