#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/recurrence/plan.hpp"
#include "../../../kernel/status.hpp"
#include "../../../resident/window/admission/runtime/windows.hpp"

#include "../../collective/chunk.hpp"
#include "../../compact/local.hpp"
#include "../../descriptor.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"
#include "../../map/source_upper.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local/state.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/capacity.hpp"
#include "../pipeline/evidence.hpp"
#include "../pipeline/recurrence.hpp"
#include "../pipeline/source.hpp"
#include "../pipeline/state.hpp"
#include "../reset_source.hpp"

#include "../../../primitive/block.hpp"
#include "../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

namespace rund::node::accel::detail {

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
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
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
      run.view_binds, run.scratch, &run, run.templates, run.failed_node,
      prepared, memory);
}

} // namespace rund::node::accel::detail
