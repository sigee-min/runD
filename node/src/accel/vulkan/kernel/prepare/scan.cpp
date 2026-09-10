#include "internal.hpp"

#include "../../partition/local.hpp"
#include "../../scan/local.hpp"
#include "../../scan/pipeline.hpp"
#include "../../segmented/local.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
AppendVulkanScanPipelines(VulkanAdapter &adapter, const BoundStep &step,
                          VulkanKernelImmutablePipelines &pipelines) {
  const auto *const active = OperationFor<operation::Scan>(step);
  const RangePrefixExec prefix = active == nullptr
                                     ? PlanRangeFlatPrefix(0u, 0u, 0u, 0u)
                                     : PlanScanPrefixExecution(active->plan);
  return active != nullptr && prefix.ok() &&
         pipelines.append(AcquireVulkanScanPipeline(adapter, active->desc,
                                                    step.planned->domain,
                                                    VulkanScanStage::Block),
                          kScanDescriptorCount, 1u) &&
         (!ScanPrefixHasOffset(prefix) ||
          (pipelines.append(AcquireVulkanScanPipeline(adapter, active->desc,
                                                      step.planned->domain,
                                                      VulkanScanStage::Prefix),
                            kScanDescriptorCount, 1u) &&
           pipelines.append(AcquireVulkanScanPipeline(adapter, active->desc,
                                                      step.planned->domain,
                                                      VulkanScanStage::Offset),
                            kScanDescriptorCount, 1u)));
}

[[nodiscard]] bool
AppendVulkanSegmentedScanPipelines(VulkanAdapter &adapter,
                                   const BoundStep &step,
                                   VulkanKernelImmutablePipelines &pipelines) {
  const auto *const active = OperationFor<operation::SegmentedScan>(step);
  return active != nullptr &&
         pipelines.append(AcquireSegmentedScanPipeline(
                              adapter, active->desc, step.planned->domain,
                              VulkanSegmentedScanStage::Block),
                          kSegmentedScanDescriptorCount, 1u) &&
         (active->plan.pass_count == 1u ||
          (pipelines.append(AcquireSegmentedScanPipeline(
                                adapter, active->desc, step.planned->domain,
                                VulkanSegmentedScanStage::Prefix),
                            kSegmentedScanDescriptorCount, 1u) &&
           pipelines.append(AcquireSegmentedScanPipeline(
                                adapter, active->desc, step.planned->domain,
                                VulkanSegmentedScanStage::Offset),
                            kSegmentedScanDescriptorCount, 1u)));
}

[[nodiscard]] bool
AppendVulkanPartitionPipelines(VulkanAdapter &adapter, const BoundStep &step,
                               VulkanKernelImmutablePipelines &pipelines) {
  const auto *const active = OperationFor<operation::Partition>(step);
  const rund::kernel::ScanDesc scan{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = active == nullptr ? 0u : active->plan.element_count,
      .block_size = block::VulkanPartition,
  };
  const rund::kernel::ScanPlan scan_plan = rund::kernel::PlanScan(scan);
  const RangePrefixExec prefix = PlanScanPrefixExecution(scan_plan);
  return active != nullptr && prefix.ok() &&
         pipelines.append(AcquirePartitionPipeline(adapter, active->desc,
                                                   PartitionStage::Classify),
                          kPartitionClassifyDescriptorCount, 1u) &&
         pipelines.append(AcquirePartitionPipeline(adapter, active->desc,
                                                   PartitionStage::Scatter),
                          kPartitionScatterDescriptorCount, 1u) &&
         pipelines.append(AcquireVulkanScanPipeline(
                              adapter, scan, rund::kernel::ComputeDomain::U32,
                              VulkanScanStage::Block),
                          kScanDescriptorCount, 1u) &&
         (!ScanPrefixHasOffset(prefix) ||
          (pipelines.append(AcquireVulkanScanPipeline(
                                adapter, scan, rund::kernel::ComputeDomain::U32,
                                VulkanScanStage::Prefix),
                            kScanDescriptorCount, 1u) &&
           pipelines.append(AcquireVulkanScanPipeline(
                                adapter, scan, rund::kernel::ComputeDomain::U32,
                                VulkanScanStage::Offset),
                            kScanDescriptorCount, 1u)));
}

} // namespace

bool MaterializeVulkanScanPipelines(VulkanAdapter &adapter,
                                    const BoundStep &step,
                                    VulkanKernelImmutablePipelines &pipelines) {
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Scan:
    return AppendVulkanScanPipelines(adapter, step, pipelines);
  case rund::kernel::NodeKind::SegmentedScan:
    return AppendVulkanSegmentedScanPipelines(adapter, step, pipelines);
  case rund::kernel::NodeKind::Partition:
    return AppendVulkanPartitionPipelines(adapter, step, pipelines);
  default:
    return false;
  }
}

#endif

} // namespace rund::node::accel::detail
