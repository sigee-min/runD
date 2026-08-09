#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/status.hpp"

#include "../../../sort/block/metal.hpp"
#include "../../buffer/owner.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../pipeline/guard.hpp"
#include "../../pipeline/source_recipe.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/source_upper.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/source.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/build.hpp"
#include "../pipeline/identity_index.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck RunMetalKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<void> prepared{};
  PreparedMemory memory{};
  const rund::AccelCheck ready = PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
  if (ready.ok && run.traffic != nullptr) {
    *run.traffic = MetalKernelTraffic(prepared);
  }
  return ready.ok ? RunMetalResources(*run.pick, prepared) : ready;
}

rund::AccelCheck PrepareMetalKernel(const BackendRun &run,
                                    std::shared_ptr<void> &prepared,
                                    PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
}

rund::AccelCheck
PrepareMetalPipelinePrivateKernel(const BackendRun &run,
                                  std::shared_ptr<void> &prepared,
                                  PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::PipelinePrivate, run.resets, run.views,
      run.view_binds, run.scratch, &run, run.templates, run.failed_node,
      prepared, memory);
}

rund::AccelCheck SubmitPreparedMetalKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    const KernelCompletion completion, void *const user,
    PreparedMemoryMeter *const memory, const std::shared_ptr<void> &,
    const KernelTiming timing) noexcept {
  return run.pick != nullptr && prepared != nullptr && completion != nullptr
             ? SubmitMetalResources(*run.pick, prepared, completion, user,
                                    memory, timing)
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

} // namespace rund::node::accel::detail
