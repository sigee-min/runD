#include "local.hpp"

#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../numeric/resource.hpp"
#include "../../numeric/source.hpp"
#include "../../partition/local.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/local.hpp"
#include "../../scan/pipeline.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local/api.hpp"

#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/scan/plan.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/stage.hpp>

#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

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

} // namespace

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
    const RangePrefixExec prefix = active == nullptr
                                       ? PlanRangeFlatPrefix(0u, 0u, 0u, 0u)
                                       : PlanScanPrefixExecution(active->plan);
    complete = active != nullptr && prefix.ok() &&
               add(AcquireVulkanScanPipeline(*adapter, active->desc,
                                             step.planned->domain,
                                             VulkanScanStage::Block),
                   kScanDescriptorCount) &&
               (!ScanPrefixHasOffset(prefix) ||
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
    const RangePrefixExec prefix = PlanScanPrefixExecution(scan_plan);
    complete = active != nullptr && prefix.ok() &&
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
               (!ScanPrefixHasOffset(prefix) ||
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
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window: {
    const RangePlan *const range =
        step.step == nullptr ? nullptr : RangePlanFor(step.step->operation);
    const std::optional<RangeExec> execution =
        range == nullptr ? std::nullopt : RangeExec::from(*range);
    complete = execution.has_value();
    if (complete) {
      const std::uint32_t descriptor_count = execution->descriptor_count();
      if (step.step->kind() == rund::kernel::NodeKind::Window &&
          range->shape().resident_counted()) {
        complete = pipelines->append_control(
            AcquireVulkanRangeControlPipeline(*adapter, *range), 4u, 1u);
      }
      VulkanCollectivePipeline *const data_pipeline =
          complete ? AcquireVulkanRangePipeline(*adapter, *execution) : nullptr;
      complete = complete && data_pipeline != nullptr;
      for (std::size_t index = 0u; index < range->stage_count(); ++index) {
        complete = complete && add(data_pipeline, descriptor_count);
      }
    }
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

#endif

} // namespace rund::node::accel::detail
