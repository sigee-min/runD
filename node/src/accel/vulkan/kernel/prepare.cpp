#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../scratch.hpp"
#include "local.hpp"
#include "ops/table.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanStep(const rund::AccelDevice &pick,
                                   const BoundStep &step,
                                   const VulkanKernelOps &ops,
                                   const KernelPreparationMode mode,
                                   std::shared_ptr<void> &resources) {
  if (ops.prepare != nullptr) {
    return ops.prepare(pick, step, mode, resources);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck PrepareVulkanSteps(const rund::AccelDevice &pick,
                                    const BoundStep *const steps,
                                    const std::size_t step_count,
                                    const KernelPreparationMode mode,
                                    std::uint32_t *const failed_node,
                                    VulkanKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || resources.size() != step_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const KernelPreparationScope preparation{mode};
  bool scratch_seen = false;
  for (std::size_t index = 0u; index < step_count; ++index) {
    VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const rund::AccelCheck view_commands =
        PrepareVulkanViewCommands(*resources.adapter, entry->view);
    if (!view_commands.ok) {
      RecordNode(failed_node, steps[index]);
      return view_commands;
    }
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    VulkanScratch *const scratch = ActiveVulkanScratch();
    if (scratch != nullptr) {
      scratch->reset();
    }
    const rund::AccelCheck prepare = PrepareVulkanStep(
        pick, prepared_step, entry->ops, mode, entry->resource);
    if (!prepare.ok) {
      RecordNode(failed_node, steps[index]);
      return prepare;
    }
    const bool scratch_used = scratch != nullptr && scratch->active();
    entry->barrier_before =
        entry->barrier_before || (scratch_seen && scratch_used);
    scratch_seen = scratch_seen || scratch_used;
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck PrepareVulkanStepViews(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const KernelPreparationMode mode,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || !resources.reserve(step_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    entry->ops = VulkanKernelOpsFor(steps[index].step->kind());
    entry->resets = steps[index].resets;
    entry->barrier_before = steps[index].barrier_before;
    const rund::AccelCheck view = PrepareVulkanViewLowering(
        pick, steps[index], mode, views, view_binds, entry->view);
    if (!view.ok) {
      RecordNode(failed_node, steps[index]);
      return view;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
