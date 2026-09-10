#include "internal.hpp"

#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local/api.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool MaterializeVulkanCollectivePipelines(
    VulkanAdapter &adapter, const BoundStep &step,
    VulkanKernelImmutablePipelines &pipelines) {
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto *const active = OperationFor<operation::SegmentedReduce>(step);
    bool complete = active != nullptr;
    for (const VulkanSegmentedReduceStage stage :
         {VulkanSegmentedReduceStage::Classify,
          VulkanSegmentedReduceStage::Prefix,
          VulkanSegmentedReduceStage::Scatter,
          VulkanSegmentedReduceStage::Reduce}) {
      complete =
          complete && pipelines.append(AcquireVulkanSegmentedReducePipeline(
                                           adapter, active->desc, active->plan,
                                           step.planned->domain, stage),
                                       kVulkanSegmentedReduceBindings, 1u);
    }
    return complete;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto *const active = OperationFor<operation::Sort>(step);
    return active != nullptr &&
           pipelines.append(
               AcquireSortPipeline(adapter, active->desc, SortStage::Dispatch),
               kSortDescriptorCount, 1u) &&
           pipelines.append(
               AcquireSortPipeline(adapter, active->desc, SortStage::Classify),
               kSortDescriptorCount, active->plan.radix_pass_count) &&
           pipelines.append(
               AcquireSortPipeline(adapter, active->desc, SortStage::Prefix),
               kSortDescriptorCount, active->plan.radix_pass_count) &&
           pipelines.append(
               AcquireSortPipeline(adapter, active->desc, SortStage::Base),
               kSortDescriptorCount, active->plan.radix_pass_count) &&
           pipelines.append(
               AcquireSortPipeline(adapter, active->desc, SortStage::Scatter),
               kSortDescriptorCount, active->plan.radix_pass_count);
  }
  case rund::kernel::NodeKind::Compact: {
    const auto *const active = OperationFor<operation::Compact>(step);
    bool complete = active != nullptr;
    for (const CompactStage stage :
         {CompactStage::Classify, CompactStage::Prefix,
          CompactStage::Scatter}) {
      complete =
          complete &&
          pipelines.append(AcquireCompactPipeline(adapter, active->desc, stage),
                           kCompactDescriptorCount, 1u);
    }
    return complete;
  }
  case rund::kernel::NodeKind::Gather: {
    const auto *const active = OperationFor<operation::Gather>(step);
    return active != nullptr &&
           pipelines.append(AcquireGatherPipeline(adapter, active->desc, true),
                            kGatherDescriptorCount, 1u) &&
           pipelines.append(AcquireGatherPipeline(adapter, active->desc, false),
                            kGatherDescriptorCount, 1u);
  }
  case rund::kernel::NodeKind::Histogram: {
    const auto *const active = OperationFor<operation::Histogram>(step);
    return active != nullptr &&
           pipelines.append(
               AcquireHistogramPipeline(adapter, active->desc, true),
               kHistogramDescriptorCount, 1u) &&
           pipelines.append(
               AcquireHistogramPipeline(adapter, active->desc, false),
               kHistogramDescriptorCount, 1u);
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto *const active = OperationFor<operation::Reduce>(step);
    return active != nullptr &&
           pipelines.append(AcquireReducePipeline(adapter, active->desc,
                                                  step.planned->domain),
                            kReduceDescriptorCount, active->plan.pass_count);
  }
  case rund::kernel::NodeKind::Scatter: {
    const auto *const active = OperationFor<operation::Scatter>(step);
    return active != nullptr &&
           pipelines.append(AcquireScatterPipeline(adapter, active->desc),
                            kScatterDescriptorCount, 1u);
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto *const active = OperationFor<operation::ScatterReduce>(step);
    bool complete = active != nullptr;
    for (const VulkanScatterReduceStage stage :
         {VulkanScatterReduceStage::Control, VulkanScatterReduceStage::Init,
          VulkanScatterReduceStage::Fold}) {
      complete =
          complete && pipelines.append(AcquireVulkanScatterReducePipeline(
                                           adapter, active->plan, stage),
                                       kVulkanScatterReduceBindings, 1u);
    }
    return complete;
  }
  default:
    return false;
  }
}

#endif

} // namespace rund::node::accel::detail
