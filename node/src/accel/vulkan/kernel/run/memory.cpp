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

#include "storage.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
ObserveVulkanMapTemplate(const VulkanMapTemplateResources &prepared,
                         std::uint64_t &bytes) noexcept {
  if (prepared.adapter == nullptr || prepared.pipeline == nullptr ||
      !prepared.plan.ok ||
      prepared.input_plans.size() != prepared.plan.input_buffer_count ||
      prepared.input_layouts.size() != prepared.plan.input_buffer_count ||
      prepared.output_layouts.size() != prepared.plan.output_buffer_count ||
      prepared.checks.size() > prepared.plan.input_buffer_count ||
      !backend_template_plan::add(bytes, sizeof(VulkanMapTemplateResources)) ||
      !AddVulkanVectorStorage(bytes, prepared.input_plans) ||
      !AddVulkanVectorStorage(bytes, prepared.input_layouts) ||
      !AddVulkanVectorStorage(bytes, prepared.output_layouts) ||
      !AddVulkanVectorStorage(bytes, prepared.checks)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool
ObserveVulkanMapDescriptorArena(const VulkanMapDescriptorArena &arena,
                                const std::uint64_t expected_sets,
                                std::uint64_t &bytes) noexcept {
  if (arena.adapter == nullptr || arena.pool == VK_NULL_HANDLE ||
      arena.sets.size() != expected_sets || arena.next > arena.sets.size() ||
      !backend_template_plan::add(bytes, sizeof(VulkanMapDescriptorArena)) ||
      !AddVulkanVectorStorage(bytes, arena.sets)) {
    return false;
  }
  for (const VkDescriptorSet set : arena.sets) {
    if (set == VK_NULL_HANDLE) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
ObserveVulkanProgramTemplate(const VulkanKernelProgramTemplate &program,
                             std::uint64_t &bytes) noexcept {
  if (program.kind != VulkanKernelTemplateKind::Program ||
      program.signature == nullptr || !program.route_demand.valid() ||
      program.signature->steps == nullptr ||
      program.steps.size() != program.signature->step_count ||
      program.steps.empty() ||
      (program.reset_set_count != 0u) != (program.reset_pipeline != nullptr) ||
      (program.view_set_count != 0u) != (program.view_pipeline != nullptr) ||
      !backend_template_plan::add(bytes, sizeof(VulkanKernelProgramTemplate)) ||
      !AddVulkanVectorStorage(bytes, program.steps) ||
      !AddVulkanVectorStorage(bytes, program.descriptor_dependencies)) {
    return false;
  }

  for (std::size_t index = 0u; index < program.steps.size(); ++index) {
    const VulkanKernelProgramStepTemplate &step = program.steps[index];
    if (step.immutable == nullptr || !step.manifest.ok ||
        program.signature->steps[index].step == nullptr) {
      return false;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (program.steps[prior].immutable.get() == step.immutable.get()) {
        first_owner = false;
        break;
      }
    }
    if (!first_owner) {
      continue;
    }
    if (step.ops.encode == EncodeVulkanMap) {
      const auto *const map =
          static_cast<const VulkanMapTemplateResources *>(step.immutable.get());
      if (map == nullptr || !ObserveVulkanMapTemplate(*map, bytes)) {
        return false;
      }
      continue;
    }
    const auto *const pipelines =
        static_cast<const VulkanKernelImmutablePipelines *>(
            step.immutable.get());
    if (pipelines == nullptr ||
        !pipelines->ready(program.signature->steps[index].step->kind(),
                          step.manifest) ||
        !backend_template_plan::add(bytes,
                                    sizeof(VulkanKernelImmutablePipelines))) {
      return false;
    }
  }

  for (std::size_t index = 0u; index < program.descriptor_dependencies.size();
       ++index) {
    const VulkanKernelDescriptorDependency &dependency =
        program.descriptor_dependencies[index];
    std::uint64_t expected_capacity = 0u;
    if (dependency.identity == nullptr || dependency.descriptor_count == 0u ||
        dependency.sets_per_route == 0u || dependency.set_capacity == 0u ||
        dependency.source_node == NoNode ||
        !rund::kernel::checked::mul(dependency.sets_per_route,
                                    program.route_demand.capacity,
                                    expected_capacity) ||
        expected_capacity != dependency.set_capacity) {
      return false;
    }
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Collective) {
      if (dependency.map_arena != nullptr) {
        return false;
      }
      continue;
    }
    if (dependency.kind != VulkanKernelDescriptorDependencyKind::Map ||
        dependency.map_arena == nullptr) {
      return false;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const VulkanKernelDescriptorDependency &previous =
          program.descriptor_dependencies[prior];
      if (previous.map_arena.get() == dependency.map_arena.get()) {
        first_owner = false;
        break;
      }
    }
    if (first_owner &&
        !ObserveVulkanMapDescriptorArena(*dependency.map_arena,
                                         dependency.set_capacity, bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
ObserveVulkanRecurrenceTemplate(const VulkanMapRecurrenceTemplate &recurrence,
                                std::uint64_t &bytes) noexcept {
  std::uint64_t expected_sets = 0u;
  if (recurrence.signature == nullptr) {
    return false;
  }
  const MapRecurrencePreparationPlan preparation = PlanMapRecurrencePreparation(
      *recurrence.signature, 1u, recurrence.history ? 1u : 0u);
  const MapRecurrenceSourcePlan &source = recurrence.history
                                              ? preparation.history_source
                                              : preparation.terminal_source;
  if (recurrence.kind != VulkanKernelTemplateKind::MapRecurrence ||
      !preparation.eligible() || !source.ok ||
      source.history != recurrence.history ||
      (recurrence.history ? preparation.history_group_count == 0u
                          : preparation.terminal_group_count() == 0u) ||
      recurrence.group_capacity == 0u || recurrence.prepared == nullptr ||
      recurrence.descriptors == nullptr ||
      recurrence.prepared->control_pipeline != nullptr ||
      recurrence.prepared->check_pipeline != nullptr ||
      !recurrence.prepared->checks.empty() ||
      !rund::kernel::checked::mul(recurrence.group_capacity,
                                  recurrence.prepared->plan.dispatch_count,
                                  expected_sets) ||
      expected_sets != recurrence.descriptor_set_capacity ||
      !backend_template_plan::add(bytes, sizeof(VulkanMapRecurrenceTemplate)) ||
      !ObserveVulkanMapTemplate(*recurrence.prepared, bytes) ||
      !ObserveVulkanMapDescriptorArena(*recurrence.descriptors, expected_sets,
                                       bytes)) {
    return false;
  }
  return recurrence.prepared->adapter == recurrence.descriptors->adapter;
}

} // namespace
#endif

PreparedMemory
ObserveVulkanPipelineTemplate(const void *const prepared) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  std::uint64_t bytes = 0u;
  bool observed = false;
  if (prepared != nullptr) {
    switch (VulkanKernelTemplateKindOf(prepared)) {
    case VulkanKernelTemplateKind::Program:
      observed = ObserveVulkanProgramTemplate(
          *static_cast<const VulkanKernelProgramTemplate *>(prepared), bytes);
      break;
    case VulkanKernelTemplateKind::MapRecurrence:
      observed = ObserveVulkanRecurrenceTemplate(
          *static_cast<const VulkanMapRecurrenceTemplate *>(prepared), bytes);
      break;
    }
  }
  if (!observed || bytes == 0u) {
    constexpr std::uint64_t invalid = std::numeric_limits<std::uint64_t>::max();
    return PreparedMemory{.current = invalid,
                          .peak = invalid,
                          .cumulative = invalid,
                          .budget = invalid};
  }
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
#else
  (void)prepared;
  return {};
#endif
}

} // namespace rund::node::accel::detail
