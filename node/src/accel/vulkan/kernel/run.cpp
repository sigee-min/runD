#include "../../kernel/backend/execute.hpp"

#include "../kernel.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck RunVulkanKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<void> prepared{};
  PreparedMemory memory{};
  const rund::AccelCheck ready = PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, run.failed_node, prepared, memory);
  if (ready.ok && run.traffic != nullptr) {
    *run.traffic = VulkanKernelTraffic(prepared);
  }
  return ready.ok ? RunVulkanResources(*run.pick, prepared) : ready;
}

rund::AccelCheck PrepareVulkanKernel(const BackendRun &run,
                                     std::shared_ptr<void> &prepared,
                                     PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, run.failed_node, prepared, memory);
}

rund::AccelCheck
PrepareVulkanPipelinePrivateKernel(const BackendRun &run,
                                   std::shared_ptr<void> &prepared,
                                   PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::PipelinePrivate, run.resets, run.views,
      run.view_binds, run.scratch, run.failed_node, prepared, memory);
}

rund::AccelCheck SubmitPreparedVulkanKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    const KernelCompletion completion, void *const user, PreparedMemoryMeter *,
    const std::shared_ptr<void> &) noexcept {
  return run.pick != nullptr && prepared != nullptr && completion != nullptr
             ? SubmitVulkanResources(*run.pick, prepared, completion, user)
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

} // namespace rund::node::accel::detail
