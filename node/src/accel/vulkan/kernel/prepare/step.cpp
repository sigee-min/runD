#include "local.hpp"

#include "../../scratch.hpp"
#include "../ops/table.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck PrepareVulkanMapRouteStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    const VulkanKernelProgramTemplate &program,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<void> &resources) {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned == nullptr ||
      step.planned->artifact == nullptr || prepared == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<VulkanMapDescriptorArena> descriptors;
  for (const VulkanKernelDescriptorDependency &dependency :
       program.descriptor_dependencies) {
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Map &&
        dependency.identity == prepared->pipeline) {
      descriptors = dependency.map_arena;
      break;
    }
  }
  if (descriptors == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  return PrepareVulkanMapRoute(
      pick, step.planned->plan, *step.planned->artifact,
      step.map_windows.data(), step.map_windows.size(), MapBindingFor(step),
      step.control, std::move(prepared), std::move(descriptors), resources);
}

} // namespace

rund::AccelCheck
PrepareVulkanStep(const rund::AccelDevice &pick, const BoundStep &step,
                  const VulkanKernelOps &ops, const KernelPreparationMode mode,
                  const VulkanKernelImmutablePipelines *const pipelines,
                  std::shared_ptr<void> &resources) {
  if (ops.prepare != nullptr) {
    return ops.prepare(pick, step, mode, pipelines, resources);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck PrepareVulkanKernelProgramTemplate(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const KernelPreparationMode mode,
    const BackendRun *const template_probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  return IsPipelinePrivatePreparation(mode)
             ? AcquireVulkanProgramTemplate(pick, steps, step_count,
                                            template_probe, templates,
                                            failed_node, resources)
             : rund::AccelCheck{true, "ok"};
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
    if (entry->view != nullptr && !entry->view->transfers.empty() &&
        resources.program != nullptr) {
      if (resources.program->view_pipeline == nullptr ||
          resources.program->view_set_count == 0u) {
        RecordNode(failed_node, steps[index]);
        return rund::AccelCheck{false, "accel_kernel_template_invalid"};
      }
      entry->view->pipeline = resources.program->view_pipeline;
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
    entry->ops = resources.program == nullptr
                     ? VulkanKernelOpsFor(steps[index].step->kind())
                     : resources.program->steps[index].ops;
    const rund::AccelCheck prepare =
        resources.program != nullptr &&
                prepared_step.step->kind() == rund::kernel::NodeKind::Map
            ? PrepareVulkanMapRouteStep(
                  pick, prepared_step, *resources.program,
                  std::static_pointer_cast<const VulkanMapTemplateResources>(
                      resources.program->steps[index].immutable),
                  entry->resource)
            : PrepareVulkanStep(
                  pick, prepared_step, entry->ops, mode,
                  resources.program == nullptr
                      ? nullptr
                      : static_cast<const VulkanKernelImmutablePipelines *>(
                            resources.program->steps[index].immutable.get()),
                  entry->resource);
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
