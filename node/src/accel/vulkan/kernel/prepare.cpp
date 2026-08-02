#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/prepared/template_registry.hpp"
#include "../compact/local.hpp"
#include "../gather/local.hpp"
#include "../histogram/local.hpp"
#include "../numeric/resource.hpp"
#include "../numeric/source.hpp"
#include "../partition/local.hpp"
#include "../reduce/local.hpp"
#include "../scan/local.hpp"
#include "../scan/pipeline.hpp"
#include "../scatter/local.hpp"
#include "../scatter/reduce/model.hpp"
#include "../scratch.hpp"
#include "../segmented/local.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/local/api.hpp"
#include "../stencil/local.hpp"
#include "local.hpp"
#include "ops/table.hpp"
#include "reset_source.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/scan/plan.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/stage.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>

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
  if (prepared == nullptr ||
      VulkanKernelTemplateKindOf(prepared) !=
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

[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanNumericStepPipeline(VulkanAdapter &adapter,
                                 const BoundStep &step) {
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Transform: {
    const auto *const active = OperationFor<operation::Transform>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash =
        rund::kernel::HashTransform(rund::kernel::TransformDesc{});
    return AcquireNumericPipeline(
        adapter, 6u, sizeof(rund::kernel::transform_stage::Batch),
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? TransformSource64() : TransformSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Matrix: {
    const auto *const active = OperationFor<operation::Matrix>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashMatrix(
        rund::kernel::MatrixDesc{.element_bytes = active->desc.element_bytes});
    const rund::kernel::ComputeDomain domain =
        active->plan.arithmetic == rund::kernel::MatrixArithmetic::Fixed
            ? rund::kernel::ComputeDomain::Fixed
            : (active->plan.arithmetic ==
                       rund::kernel::MatrixArithmetic::SignedWrap
                   ? (wide ? rund::kernel::ComputeDomain::I64
                           : rund::kernel::ComputeDomain::I32)
                   : (wide ? rund::kernel::ComputeDomain::U64
                           : rund::kernel::ComputeDomain::U32));
    return AcquireNumericPipeline(
        adapter, 4u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          domain,
                          active->plan.arithmetic ==
                                  rund::kernel::MatrixArithmetic::Fixed
                              ? active->plan.fixed_format
                              : rund::kernel::ComputeFixedFormat{}),
        wide ? MatrixSource64() : MatrixSource(),
        MatrixPolicy(active->plan.arithmetic, active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Factor: {
    const auto *const active = OperationFor<operation::Factor>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashFactor(
        rund::kernel::FactorDesc{.element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 5u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? FactorSource64() : FactorSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Solve: {
    const auto *const active = OperationFor<operation::Solve>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashSolve(
        rund::kernel::SolveDesc{.element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 6u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? SolveSource64() : SolveSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Spectrum: {
    const auto *const active = OperationFor<operation::Spectrum>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashSpectrum(rund::kernel::SpectrumDesc{
        .element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 5u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? SpectrumSource64() : SpectrumSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  default:
    return nullptr;
  }
}

[[nodiscard]] rund::AccelCheck MaterializeVulkanPrimitivePipelines(
    const rund::AccelDevice &pick, const BoundStep &step,
    const PreparedBackendManifest &manifest,
    std::shared_ptr<const VulkanKernelImmutablePipelines> &out) {
  out.reset();
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || step.step == nullptr || step.planned == nullptr ||
      step.step->kind() == rund::kernel::NodeKind::Map || !manifest.ok) {
    return {false, "accel_kernel_template_invalid"};
  }
  std::shared_ptr<VulkanKernelImmutablePipelines> pipelines;
  try {
    pipelines = std::make_shared<VulkanKernelImmutablePipelines>();
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_capacity"};
  }
  pipelines->kind = step.step->kind();
  const auto add = [&](VulkanCollectivePipeline *const pipeline,
                       const std::uint32_t descriptors,
                       const std::uint64_t sets = 1u) {
    return pipelines->append(pipeline, descriptors, sets);
  };
  bool complete = false;
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Scan: {
    const auto *const active = OperationFor<operation::Scan>(step);
    complete = active != nullptr &&
               add(AcquireVulkanScanPipeline(*adapter, active->desc,
                                             step.planned->domain,
                                             VulkanScanStage::Block),
                   kScanDescriptorCount) &&
               (active->plan.pass_count == 1u ||
                (add(AcquireVulkanScanPipeline(*adapter, active->desc,
                                               step.planned->domain,
                                               VulkanScanStage::Prefix),
                     kScanDescriptorCount) &&
                 add(AcquireVulkanScanPipeline(*adapter, active->desc,
                                               step.planned->domain,
                                               VulkanScanStage::Offset),
                     kScanDescriptorCount)));
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto *const active = OperationFor<operation::SegmentedScan>(step);
    complete =
        active != nullptr &&
        add(AcquireSegmentedScanPipeline(*adapter, active->desc,
                                         step.planned->domain,
                                         VulkanSegmentedScanStage::Block),
            kSegmentedScanDescriptorCount) &&
        (active->plan.pass_count == 1u ||
         (add(AcquireSegmentedScanPipeline(*adapter, active->desc,
                                           step.planned->domain,
                                           VulkanSegmentedScanStage::Prefix),
              kSegmentedScanDescriptorCount) &&
          add(AcquireSegmentedScanPipeline(*adapter, active->desc,
                                           step.planned->domain,
                                           VulkanSegmentedScanStage::Offset),
              kSegmentedScanDescriptorCount)));
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto *const active = OperationFor<operation::SegmentedReduce>(step);
    complete = active != nullptr;
    for (const VulkanSegmentedReduceStage stage :
         {VulkanSegmentedReduceStage::Classify,
          VulkanSegmentedReduceStage::Prefix,
          VulkanSegmentedReduceStage::Scatter,
          VulkanSegmentedReduceStage::Reduce}) {
      complete = complete && add(AcquireVulkanSegmentedReducePipeline(
                                     *adapter, active->desc, active->plan,
                                     step.planned->domain, stage),
                                 kVulkanSegmentedReduceBindings);
    }
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto *const active = OperationFor<operation::Sort>(step);
    complete =
        active != nullptr &&
        add(AcquireSortPipeline(*adapter, active->desc, SortStage::Dispatch),
            kSortDescriptorCount) &&
        add(AcquireSortPipeline(*adapter, active->desc, SortStage::Classify),
            kSortDescriptorCount, active->plan.radix_pass_count) &&
        add(AcquireSortPipeline(*adapter, active->desc, SortStage::Prefix),
            kSortDescriptorCount, active->plan.radix_pass_count) &&
        add(AcquireSortPipeline(*adapter, active->desc, SortStage::Base),
            kSortDescriptorCount, active->plan.radix_pass_count) &&
        add(AcquireSortPipeline(*adapter, active->desc, SortStage::Scatter),
            kSortDescriptorCount, active->plan.radix_pass_count);
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    const auto *const active = OperationFor<operation::Compact>(step);
    complete = active != nullptr;
    for (const CompactStage stage :
         {CompactStage::Classify, CompactStage::Prefix,
          CompactStage::Scatter}) {
      complete =
          complete && add(AcquireCompactPipeline(*adapter, active->desc, stage),
                          kCompactDescriptorCount);
    }
    break;
  }
  case rund::kernel::NodeKind::Gather: {
    const auto *const active = OperationFor<operation::Gather>(step);
    complete = active != nullptr &&
               add(AcquireGatherPipeline(*adapter, active->desc, true),
                   kGatherDescriptorCount) &&
               add(AcquireGatherPipeline(*adapter, active->desc, false),
                   kGatherDescriptorCount);
    break;
  }
  case rund::kernel::NodeKind::Histogram: {
    const auto *const active = OperationFor<operation::Histogram>(step);
    complete = active != nullptr &&
               add(AcquireHistogramPipeline(*adapter, active->desc, true),
                   kHistogramDescriptorCount) &&
               add(AcquireHistogramPipeline(*adapter, active->desc, false),
                   kHistogramDescriptorCount);
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto *const active = OperationFor<operation::Partition>(step);
    const rund::kernel::ScanDesc scan{
        .op = rund::kernel::ScanOp::ExclusiveSum,
        .element = rund::kernel::ScanElement::U32,
        .element_count = active == nullptr ? 0u : active->plan.element_count,
        .block_size = block::VulkanPartition,
    };
    const rund::kernel::ScanPlan scan_plan = rund::kernel::PlanScan(scan);
    complete = active != nullptr && scan_plan.ok &&
               add(AcquirePartitionPipeline(*adapter, active->desc,
                                            PartitionStage::Classify),
                   kPartitionClassifyDescriptorCount) &&
               add(AcquirePartitionPipeline(*adapter, active->desc,
                                            PartitionStage::Scatter),
                   kPartitionScatterDescriptorCount) &&
               add(AcquireVulkanScanPipeline(*adapter, scan,
                                             rund::kernel::ComputeDomain::U32,
                                             VulkanScanStage::Block),
                   kScanDescriptorCount) &&
               (scan_plan.pass_count == 1u ||
                (add(AcquireVulkanScanPipeline(*adapter, scan,
                                               rund::kernel::ComputeDomain::U32,
                                               VulkanScanStage::Prefix),
                     kScanDescriptorCount) &&
                 add(AcquireVulkanScanPipeline(*adapter, scan,
                                               rund::kernel::ComputeDomain::U32,
                                               VulkanScanStage::Offset),
                     kScanDescriptorCount)));
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto *const active = OperationFor<operation::Reduce>(step);
    complete =
        active != nullptr &&
        add(AcquireReducePipeline(*adapter, active->desc, step.planned->domain),
            kReduceDescriptorCount, active->plan.pass_count);
    break;
  }
  case rund::kernel::NodeKind::Scatter: {
    const auto *const active = OperationFor<operation::Scatter>(step);
    complete =
        active != nullptr && add(AcquireScatterPipeline(*adapter, active->desc),
                                 kScatterDescriptorCount);
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto *const active = OperationFor<operation::ScatterReduce>(step);
    complete = active != nullptr;
    for (const VulkanScatterReduceStage stage :
         {VulkanScatterReduceStage::Control, VulkanScatterReduceStage::Init,
          VulkanScatterReduceStage::Fold}) {
      complete =
          complete &&
          add(AcquireVulkanScatterReducePipeline(*adapter, active->plan, stage),
              kVulkanScatterReduceBindings);
    }
    break;
  }
  case rund::kernel::NodeKind::Stencil: {
    const auto *const active = OperationFor<operation::Stencil>(step);
    complete =
        active != nullptr && add(AcquireStencilPipeline(*adapter, active->desc,
                                                        step.planned->domain),
                                 kStencilDescriptorCount);
    break;
  }
  case rund::kernel::NodeKind::Transform:
    complete = add(AcquireVulkanNumericStepPipeline(*adapter, step), 6u);
    break;
  case rund::kernel::NodeKind::Matrix:
    complete = add(AcquireVulkanNumericStepPipeline(*adapter, step), 4u);
    break;
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Spectrum:
    complete = add(AcquireVulkanNumericStepPipeline(*adapter, step), 5u);
    break;
  case rund::kernel::NodeKind::Solve:
    complete = add(AcquireVulkanNumericStepPipeline(*adapter, step), 6u);
    break;
  case rund::kernel::NodeKind::Map:
    break;
  }
  pipelines->capture_direct_dispatch_count =
      manifest.capture_direct_dispatch_count;
  pipelines->capture_indirect_dispatch_count =
      manifest.capture_indirect_dispatch_count;
  if (!complete || !pipelines->ready(step.step->kind(), manifest)) {
    return {false, VulkanLastError(adapter)};
  }
  out = std::move(pipelines);
  return {true, "ok"};
}

[[nodiscard]] bool AppendVulkanDescriptorDependency(
    VulkanKernelProgramTemplate &program,
    const VulkanKernelDescriptorDependencyKind kind, const void *const identity,
    const std::uint32_t descriptor_count, const std::uint64_t sets_per_route,
    const std::uint32_t source_node, std::uint32_t &order) noexcept {
  if (identity == nullptr || descriptor_count == 0u || sets_per_route == 0u ||
      source_node == NoNode ||
      order == std::numeric_limits<std::uint32_t>::max() ||
      program.descriptor_dependencies.size() ==
          program.descriptor_dependencies.capacity()) {
    return false;
  }
  program.descriptor_dependencies.push_back(VulkanKernelDescriptorDependency{
      .kind = kind,
      .identity = identity,
      .descriptor_count = descriptor_count,
      .sets_per_route = sets_per_route,
      .earliest_order = order++,
      .source_node = source_node,
  });
  return true;
}

[[nodiscard]] bool AppendVulkanMapDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanMapTemplateResources &prepared, const std::uint32_t source_node,
    std::uint32_t &order) noexcept {
  std::uint64_t descriptor_count = 0u;
  if (prepared.pipeline == nullptr ||
      !rund::kernel::checked::add(prepared.pipeline->input_buffer_count,
                                  prepared.pipeline->output_buffer_count,
                                  descriptor_count) ||
      !rund::kernel::checked::add(descriptor_count, 1u, descriptor_count) ||
      descriptor_count > std::numeric_limits<std::uint32_t>::max() ||
      !AppendVulkanDescriptorDependency(
          program, VulkanKernelDescriptorDependencyKind::Map, prepared.pipeline,
          static_cast<std::uint32_t>(descriptor_count),
          prepared.plan.dispatch_count, source_node, order)) {
    return false;
  }
  if (prepared.control_pipeline != nullptr &&
      !AppendVulkanDescriptorDependency(
          program, VulkanKernelDescriptorDependencyKind::Collective,
          prepared.control_pipeline,
          prepared.control_pipeline->descriptor_count, 1u, source_node,
          order)) {
    return false;
  }
  return prepared.check_pipeline == nullptr ||
         AppendVulkanDescriptorDependency(
             program, VulkanKernelDescriptorDependencyKind::Collective,
             prepared.check_pipeline, prepared.check_pipeline->descriptor_count,
             1u, source_node, order);
}

[[nodiscard]] bool AppendVulkanCollectiveDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanKernelImmutablePipelines &pipelines,
    const PreparedBackendManifest &manifest, const std::uint32_t source_node,
    std::uint32_t &order) noexcept {
  if (!pipelines.ready(pipelines.kind, manifest)) {
    return false;
  }
  for (std::size_t index = 0u; index < pipelines.count; ++index) {
    const VulkanKernelImmutablePipelineStage &stage = pipelines.stages[index];
    if (!AppendVulkanDescriptorDependency(
            program, VulkanKernelDescriptorDependencyKind::Collective,
            stage.pipeline, stage.descriptor_count, stage.sets_per_route,
            source_node, order)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SameVulkanDescriptorDependencyKey(
    const VulkanKernelDescriptorDependency &left,
    const VulkanKernelDescriptorDependency &right) noexcept {
  return left.kind == right.kind && left.identity == right.identity &&
         left.descriptor_count == right.descriptor_count;
}

[[nodiscard]] rund::AccelCheck
FinalizeVulkanDescriptorDependencies(VulkanAdapter &adapter,
                                     VulkanKernelProgramTemplate &program,
                                     std::uint32_t *const failed_node) {
  const auto fail = [&](const VulkanKernelDescriptorDependency *dependency,
                        const char *const reason) {
    if (failed_node != nullptr && *failed_node == NoNode &&
        dependency != nullptr && dependency->source_node != NoNode) {
      *failed_node = dependency->source_node;
    }
    return rund::AccelCheck{false, reason};
  };
  if (!program.route_demand.valid() ||
      program.descriptor_dependencies.empty()) {
    return fail(nullptr, "accel_kernel_template_invalid");
  }
  auto &dependencies = program.descriptor_dependencies;
  const auto key_less = [](const VulkanKernelDescriptorDependency &left,
                           const VulkanKernelDescriptorDependency &right) {
    if (left.kind != right.kind) {
      return left.kind < right.kind;
    }
    if (left.identity != right.identity) {
      return std::less<const void *>{}(left.identity, right.identity);
    }
    if (left.descriptor_count != right.descriptor_count) {
      return left.descriptor_count < right.descriptor_count;
    }
    return left.earliest_order < right.earliest_order;
  };
  std::sort(dependencies.begin(), dependencies.end(), key_less);

  std::size_t group_count = 0u;
  for (std::size_t index = 0u; index < dependencies.size(); ++index) {
    const VulkanKernelDescriptorDependency dependency = dependencies[index];
    if (index != 0u && dependencies[index - 1u].kind == dependency.kind &&
        dependencies[index - 1u].identity == dependency.identity &&
        dependencies[index - 1u].descriptor_count !=
            dependency.descriptor_count) {
      const VulkanKernelDescriptorDependency &previous =
          dependencies[index - 1u];
      return fail(previous.earliest_order <= dependency.earliest_order
                      ? &previous
                      : &dependency,
                  "accel_vulkan_descriptor_unavailable");
    }
    if (group_count == 0u || !SameVulkanDescriptorDependencyKey(
                                 dependencies[group_count - 1u], dependency)) {
      if (group_count != index) {
        dependencies[group_count] = dependency;
      }
      ++group_count;
      continue;
    }
    VulkanKernelDescriptorDependency &group = dependencies[group_count - 1u];
    if (!rund::kernel::checked::add(group.sets_per_route,
                                    dependency.sets_per_route,
                                    group.sets_per_route)) {
      return fail(&dependency, "compute_pipeline_capacity");
    }
    if (dependency.earliest_order < group.earliest_order) {
      group.earliest_order = dependency.earliest_order;
      group.source_node = dependency.source_node;
    }
  }
  dependencies.resize(group_count);
  std::sort(dependencies.begin(), dependencies.end(),
            [](const VulkanKernelDescriptorDependency &left,
               const VulkanKernelDescriptorDependency &right) {
              return left.earliest_order < right.earliest_order;
            });

  // Prove every complete group before the first native reservation. A tuple
  // mismatch or overflow can therefore never admit only a canonical prefix.
  for (VulkanKernelDescriptorDependency &dependency : dependencies) {
    std::uint64_t capacity = 0u;
    if (!rund::kernel::checked::mul(dependency.sets_per_route,
                                    program.route_demand.capacity, capacity) ||
        capacity == 0u ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      return fail(&dependency, "accel_vulkan_descriptor_unavailable");
    }
    dependency.set_capacity = capacity;
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Collective) {
      const auto *const pipeline =
          static_cast<const VulkanCollectivePipeline *>(dependency.identity);
      if (pipeline == nullptr ||
          pipeline->descriptor_count != dependency.descriptor_count) {
        return fail(&dependency, "accel_vulkan_descriptor_unavailable");
      }
      continue;
    }
    const auto *const pipeline =
        static_cast<const VulkanCachedPipeline *>(dependency.identity);
    std::uint64_t descriptor_count = 0u;
    if (pipeline == nullptr ||
        !rund::kernel::checked::add(pipeline->input_buffer_count,
                                    pipeline->output_buffer_count,
                                    descriptor_count) ||
        !rund::kernel::checked::add(descriptor_count, 1u, descriptor_count) ||
        descriptor_count != dependency.descriptor_count) {
      return fail(&dependency, "accel_vulkan_descriptor_unavailable");
    }
  }

  for (VulkanKernelDescriptorDependency &dependency : dependencies) {
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Map) {
      if (!PrepareVulkanMapDescriptorArena(
              adapter,
              *static_cast<const VulkanCachedPipeline *>(dependency.identity),
              dependency.set_capacity, dependency.map_arena)) {
        return fail(&dependency, VulkanLastError(&adapter));
      }
      continue;
    }
    auto *const pipeline = const_cast<VulkanCollectivePipeline *>(
        static_cast<const VulkanCollectivePipeline *>(dependency.identity));
    if (!ReserveVulkanCollectiveDescriptorDemand(adapter, *pipeline,
                                                 dependency.descriptor_count,
                                                 dependency.set_capacity)) {
      return fail(&dependency, VulkanLastError(&adapter));
    }
  }
  return rund::AccelCheck{true, "ok"};
}

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
  } catch (const std::bad_alloc &) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
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
  } catch (const std::bad_alloc &) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
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
