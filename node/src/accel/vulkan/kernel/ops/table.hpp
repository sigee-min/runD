#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "table/numeric.hpp"

[[nodiscard]] bool
ObserveVulkanMapFailure(const std::shared_ptr<void> &resources,
                        std::uint64_t &ordinal) noexcept;
[[nodiscard]] bool
ObserveVulkanGatherFailure(const std::shared_ptr<void> &resources,
                           std::uint64_t &ordinal) noexcept;

[[nodiscard]] inline rund::AccelCheck
NoVulkanPipelineStatus(const std::shared_ptr<void> &,
                       VulkanPipelineStatusSource &source) {
  source = {};
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
DescribeVulkanMapPipelineStatus(const std::shared_ptr<void> &,
                                VulkanPipelineStatusSource &);

[[nodiscard]] rund::AccelCheck
DescribeVulkanScanPipelineStatus(const std::shared_ptr<void> &,
                                 VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanScanPipelineTelemetry(const std::shared_ptr<void> &,
                                    VulkanPipelineTelemetrySource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanSegmentedPipelineStatus(const std::shared_ptr<void> &,
                                      VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanSegmentedReducePipelineStatus(const std::shared_ptr<void> &,
                                            VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanSortPipelineStatus(const std::shared_ptr<void> &,
                                 VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanSortPipelineTelemetry(const std::shared_ptr<void> &,
                                    VulkanPipelineTelemetrySource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanCompactPipelineStatus(const std::shared_ptr<void> &,
                                    VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanGatherPipelineStatus(const std::shared_ptr<void> &,
                                   VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanGatherPipelineTelemetry(const std::shared_ptr<void> &,
                                      VulkanPipelineTelemetrySource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanHistogramPipelineStatus(const std::shared_ptr<void> &,
                                      VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanPartitionPipelineStatus(const std::shared_ptr<void> &,
                                      VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanReducePipelineStatus(const std::shared_ptr<void> &,
                                   VulkanPipelineStatusSource &);
[[nodiscard]] rund::AccelCheck
DescribeVulkanScatterPipelineStatus(const std::shared_ptr<void> &,
                                    VulkanPipelineStatusSource &);

[[nodiscard]] inline VulkanKernelOps
VulkanKernelOpsFor(const rund::kernel::NodeKind kind) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Map:
    return {PrepareVulkanMapStep,
            EncodeVulkanMap,
            FinishVulkanMap,
            DescribeVulkanMapPipelineStatus,
            DescribeVulkanMapPipelineTelemetry,
            ObserveVulkanMapFailure};
  case rund::kernel::NodeKind::Scan:
    return {PrepareVulkanScanStep, EncodeVulkanScan, FinishVulkanScan,
            DescribeVulkanScanPipelineStatus,
            DescribeVulkanScanPipelineTelemetry};
  case rund::kernel::NodeKind::SegmentedScan:
    return {PrepareVulkanSegmentedStep, EncodeVulkanSegmentedScan,
            FinishVulkanSegmentedScan, DescribeVulkanSegmentedPipelineStatus};
  case rund::kernel::NodeKind::SegmentedReduce:
    return {PrepareVulkanSegmentedReduceStep, EncodeVulkanSegmentedReduce,
            FinishVulkanSegmentedReduce,
            DescribeVulkanSegmentedReducePipelineStatus};
  case rund::kernel::NodeKind::Sort:
    return {PrepareVulkanSortStep, EncodeVulkanSort, FinishVulkanSort,
            DescribeVulkanSortPipelineStatus,
            DescribeVulkanSortPipelineTelemetry};
  case rund::kernel::NodeKind::Compact:
    return {PrepareVulkanCompactStep, EncodeVulkanCompact, FinishVulkanCompact,
            DescribeVulkanCompactPipelineStatus};
  case rund::kernel::NodeKind::Gather:
    return {PrepareVulkanGatherStep,
            EncodeVulkanGather,
            FinishVulkanGather,
            DescribeVulkanGatherPipelineStatus,
            DescribeVulkanGatherPipelineTelemetry,
            ObserveVulkanGatherFailure};
  case rund::kernel::NodeKind::Histogram:
    return {PrepareVulkanHistogramStep, EncodeVulkanHistogram,
            FinishVulkanHistogram, DescribeVulkanHistogramPipelineStatus};
  case rund::kernel::NodeKind::Partition:
    return {PrepareVulkanPartitionStep, EncodeVulkanPartition,
            FinishVulkanPartition, DescribeVulkanPartitionPipelineStatus};
  case rund::kernel::NodeKind::Reduce:
    return {PrepareVulkanReduceStep, EncodeVulkanReduce, FinishVulkanReduce,
            DescribeVulkanReducePipelineStatus};
  case rund::kernel::NodeKind::Scatter:
    return {PrepareVulkanScatterStep, EncodeVulkanScatter, FinishVulkanScatter,
            DescribeVulkanScatterPipelineStatus};
  case rund::kernel::NodeKind::ScatterReduce:
    return {PrepareVulkanScatterReduceStep, EncodeVulkanScatterReduce,
            FinishVulkanScatterReduce,
            DescribeVulkanScatterReducePipelineStatus,
            DescribeVulkanScatterReducePipelineTelemetry};
  case rund::kernel::NodeKind::Stencil:
    return {PrepareVulkanStencilStep, EncodeVulkanStencil, FinishVulkanStencil,
            NoVulkanPipelineStatus};
  default:
    return VulkanNumericOpsFor(kind);
  }
}

#endif

} // namespace rund::node::accel::detail
