#include "local.hpp"

#include "../../../kernel/backend/exception.hpp"
#include "../../../kernel/prepared/template_registry.hpp"
#include "../../map/local.hpp"
#include "../ops/table.hpp"
#include "../reset_source.hpp"

#include <kernel/core/checked.hpp>

#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] constexpr bool
SameVulkanTemplateRouteDemand(const BackendTemplateRouteDemand left,
                              const BackendTemplateRouteDemand right) noexcept {
  return left.owner_count == right.owner_count &&
         left.route_copies == right.route_copies &&
         left.capacity == right.capacity;
}

[[nodiscard]] std::uint64_t
VulkanResetSetCount(const BackendRun &run) noexcept {
  return run.resets == nullptr ? 0u : run.resets->size();
}

[[nodiscard]] bool
DescribePreparedVulkanViewSetCount(const VulkanKernelResources &resources,
                                   std::uint64_t &count,
                                   std::uint32_t &source_node) noexcept {
  count = 0u;
  source_node = NoNode;
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    const std::uint64_t entry_count =
        entry == nullptr ? 0u : VulkanViewDispatchCount(entry->view);
    if (entry == nullptr ||
        !rund::kernel::checked::add(count, entry_count, count)) {
      return false;
    }
    if (entry_count != 0u && source_node == NoNode &&
        entry->view->step.step != nullptr) {
      source_node = entry->view->step.step->source.begin.index;
    }
  }
  return (count == 0u) == (source_node == NoNode);
}

[[nodiscard]] bool
MatchVulkanProgramTemplate(const void *const prepared,
                           const void *const probe) noexcept {
  if (prepared == nullptr || VulkanKernelTemplateKindOf(prepared) !=
                                 VulkanKernelTemplateKind::Program) {
    return false;
  }
  const auto *const program =
      static_cast<const VulkanKernelProgramTemplate *>(prepared);
  const auto *const run = static_cast<const BackendRun *>(probe);
  const std::uint64_t alignment = run == nullptr || run->pick == nullptr
                                      ? 0u
                                      : run->pick->caps.storage_alignment;
  return program != nullptr && program->signature != nullptr &&
         run != nullptr && program->route_demand.valid() &&
         run->template_route_demand.valid() &&
         SameVulkanTemplateRouteDemand(program->route_demand,
                                       run->template_route_demand) &&
         program->reset_set_count == VulkanResetSetCount(*run) &&
         (program->reset_pipeline != nullptr) ==
             (program->reset_set_count != 0u) &&
         (program->view_pipeline != nullptr) ==
             (program->view_set_count != 0u) &&
         alignment != 0u &&
         backend_template_plan::same_template(*program->signature, *run,
                                              alignment);
}

[[nodiscard]] rund::AccelCheck PrepareVulkanMapTemplateStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned == nullptr ||
      step.planned->artifact == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanMapTemplate(
      pick, step.planned->plan, *step.planned->artifact,
      step.map_windows.data(), step.map_windows.size(), MapBindingFor(step),
      step.control, prepared);
}

} // namespace

[[nodiscard]] rund::AccelCheck AcquireVulkanProgramTemplate(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const BackendRun *const probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || probe == nullptr ||
      probe->steps == nullptr || probe->step_count != step_count ||
      probe->steps[0].step == nullptr || probe->ops == nullptr ||
      !probe->template_route_demand.valid()) {
    if (steps != nullptr && step_count != 0u) {
      RecordNode(failed_node, steps[0]);
    }
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  std::uint64_t view_set_count = 0u;
  std::uint32_t view_source_node = NoNode;
  if (!DescribePreparedVulkanViewSetCount(resources, view_set_count,
                                          view_source_node)) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const KernelExecutionStep *const authority = probe->steps[0].step;
  const std::uint64_t variant_hi = probe->execution == nullptr
                                       ? step_count
                                       : probe->execution->admission.kernel_id;
  const std::uint64_t variant_lo =
      (static_cast<std::uint64_t>(step_count) << 32u) ^
      probe->original_dispatch_count ^ probe->final_dispatch_count ^
      VulkanResetSetCount(*probe) ^ (view_set_count << 1u);
  if (templates != nullptr) {
    std::shared_ptr<void> found = FindPreparedKernelTemplate(
        *templates, authority, variant_hi, variant_lo,
        MatchVulkanProgramTemplate, probe);
    if (found != nullptr) {
      resources.program =
          std::static_pointer_cast<VulkanKernelProgramTemplate>(found);
      if (resources.program->view_set_count != view_set_count) {
        RecordNode(failed_node, steps[0]);
        resources.program.reset();
        return rund::AccelCheck{false, "accel_kernel_template_invalid"};
      }
      return rund::AccelCheck{true, "ok"};
    }
  }

  std::shared_ptr<VulkanKernelProgramTemplate> program;
  try {
    program = std::make_shared<VulkanKernelProgramTemplate>();
    program->steps.resize(step_count);
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  program->signature = probe;
  program->route_demand = probe->template_route_demand;
  program->reset_set_count = resources.resets.size();
  program->view_set_count = view_set_count;
  std::uint64_t descriptor_dependency_capacity =
      (resources.resets.empty() ? 0u : 1u) + (view_set_count == 0u ? 0u : 1u);
  for (std::size_t index = 0u; index < step_count; ++index) {
    VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    VulkanKernelProgramStepTemplate &template_step = program->steps[index];
    template_step.ops = VulkanKernelOpsFor(steps[index].step->kind());
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    if (prepared_step.step == nullptr || prepared_step.planned == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    template_step.manifest = BuildVulkanBackendManifest(
        *prepared_step.step, prepared_step.planned->plan, &prepared_step,
        resources.adapter == nullptr ? 0u
                                     : resources.adapter->max_dispatch_groups);
    if (!template_step.manifest.ok) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, template_step.manifest.reason};
    }
    if (!rund::kernel::checked::add(
            descriptor_dependency_capacity,
            template_step.manifest.descriptor_dependency_count,
            descriptor_dependency_capacity)) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (descriptor_dependency_capacity >
      std::numeric_limits<std::size_t>::max()) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  try {
    program->descriptor_dependencies.reserve(
        static_cast<std::size_t>(descriptor_dependency_capacity));
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  std::uint32_t dependency_order = 0u;
  if (!resources.resets.empty()) {
    VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
    const rund::kernel::LoweringArtifact artifact = VulkanResetArtifact();
    if (adapter == nullptr || !artifact.ok) {
      RecordNode(failed_node, steps[0]);
      return rund::AccelCheck{false, artifact.ok ? "accel_vulkan_unavailable"
                                                 : artifact.reason};
    }
    program->reset_pipeline = AcquireVulkanCollectivePipeline(
        *adapter, 1u, sizeof(reset::Params), VulkanResetPlan(), artifact);
    if (program->reset_pipeline == nullptr ||
        !AppendVulkanDescriptorDependency(
            *program, VulkanKernelDescriptorDependencyKind::Collective,
            program->reset_pipeline, 1u, resources.resets.size(),
            steps[0].step->source.begin.index, dependency_order)) {
      RecordNode(failed_node, steps[0]);
      return rund::AccelCheck{false, program->reset_pipeline == nullptr
                                         ? VulkanLastError(adapter)
                                         : "compute_pipeline_capacity"};
    }
  }
  if (view_set_count != 0u) {
    VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
    program->view_pipeline =
        adapter == nullptr ? nullptr : AcquireVulkanViewPipeline(*adapter);
    if (program->view_pipeline == nullptr || view_source_node == NoNode ||
        !AppendVulkanDescriptorDependency(
            *program, VulkanKernelDescriptorDependencyKind::Collective,
            program->view_pipeline, 2u, view_set_count, view_source_node,
            dependency_order)) {
      if (failed_node != nullptr && *failed_node == NoNode &&
          view_source_node != NoNode) {
        *failed_node = view_source_node;
      }
      return rund::AccelCheck{false, program->view_pipeline == nullptr
                                         ? (adapter == nullptr
                                                ? "accel_vulkan_unavailable"
                                                : VulkanLastError(adapter))
                                         : "compute_pipeline_capacity"};
    }
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    VulkanKernelEntry *const entry = resources.entry(index);
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    VulkanKernelProgramStepTemplate &template_step = program->steps[index];
    if (prepared_step.step->kind() == rund::kernel::NodeKind::Map) {
      std::shared_ptr<const VulkanMapTemplateResources> prepared_map;
      const rund::AccelCheck ready =
          PrepareVulkanMapTemplateStep(pick, prepared_step, prepared_map);
      if (!ready.ok) {
        RecordNode(failed_node, prepared_step);
        return ready;
      }
      if (prepared_map == nullptr ||
          !AppendVulkanMapDescriptorDependencies(
              *program, *prepared_map, prepared_step.step->source.begin.index,
              dependency_order)) {
        RecordNode(failed_node, prepared_step);
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      template_step.immutable = std::move(prepared_map);
      continue;
    }
    std::shared_ptr<const VulkanKernelImmutablePipelines> pipelines;
    const rund::AccelCheck ready = MaterializeVulkanPrimitivePipelines(
        pick, prepared_step, template_step.manifest, pipelines);
    if (!ready.ok) {
      RecordNode(failed_node, prepared_step);
      return ready;
    }
    if (pipelines == nullptr ||
        !AppendVulkanCollectiveDescriptorDependencies(
            *program, *pipelines, template_step.manifest,
            prepared_step.step->source.begin.index, dependency_order)) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    template_step.immutable = std::move(pipelines);
  }

  if (!program->descriptor_dependencies.empty()) {
    auto *const adapter = CheckedVulkanAdapter(pick);
    if (adapter == nullptr) {
      RecordNode(failed_node, steps[0]);
      return rund::AccelCheck{false, "accel_vulkan_unavailable"};
    }
    const rund::AccelCheck reserved =
        FinalizeVulkanDescriptorDependencies(*adapter, *program, failed_node);
    if (!reserved.ok) {
      return reserved;
    }
  }

  std::shared_ptr<void> published = program;
  if (templates != nullptr) {
    const rund::AccelCheck stored = PublishPreparedKernelTemplate(
        *templates, authority, variant_hi, variant_lo, *probe->ops,
        MatchVulkanProgramTemplate, probe, published);
    if (!stored.ok) {
      RecordNode(failed_node, steps[0]);
      return stored;
    }
    program = std::static_pointer_cast<VulkanKernelProgramTemplate>(published);
  }
  resources.program = std::move(program);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
