#pragma once

#include "memory.hpp"
#include "prepare.hpp"
#include "status.hpp"

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#include "table/numeric.hpp"

[[nodiscard]] bool
ObserveMetalMapFailure(const std::shared_ptr<void> &resources,
                       std::uint64_t &ordinal) noexcept;
[[nodiscard]] bool
ObserveMetalGatherFailure(const std::shared_ptr<void> &resources,
                          std::uint64_t &ordinal) noexcept;

[[nodiscard]] inline MetalKernelOps
MetalKernelOpsFor(const rund::kernel::NodeKind kind) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Map:
    return {PrepareMetalMapStep,
            EncodeMetalMap,
            FinishMetalMap,
            MetalMapStepMemory,
            DescribeMetalMapPipelineStatus,
            DescribeMetalMapPipelineTelemetry,
            ObserveMetalMapFailure};
  case rund::kernel::NodeKind::Scan:
    return {PrepareMetalScanStep,
            EncodeMetalScan,
            FinishMetalScan,
            MetalScanStepMemory,
            DescribeMetalScanPipelineStatus,
            DescribeMetalScanPipelineTelemetry};
  case rund::kernel::NodeKind::SegmentedScan:
    return {PrepareMetalSegmentedStep, EncodeMetalSegmentedScan,
            FinishMetalSegmentedScan, MetalSegmentedStepMemory,
            DescribeMetalSegmentedPipelineStatus};
  case rund::kernel::NodeKind::SegmentedReduce:
    return {PrepareMetalSegmentedReduceStep, EncodeMetalSegmentedReduce,
            FinishMetalSegmentedReduce, MetalSegmentedReduceMemory,
            DescribeMetalSegmentedReducePipelineStatus};
  case rund::kernel::NodeKind::Sort:
    return {PrepareMetalSortStep,
            EncodeMetalSort,
            FinishMetalSort,
            MetalSortStepMemory,
            DescribeMetalSortPipelineStatus,
            DescribeMetalSortPipelineTelemetry};
  case rund::kernel::NodeKind::Compact:
    return {PrepareMetalCompactStep, EncodeMetalCompact, FinishMetalCompact,
            MetalCompactStepMemory, DescribeMetalCompactPipelineStatus};
  case rund::kernel::NodeKind::Gather:
    return {PrepareMetalGatherStep,
            EncodeMetalGather,
            FinishMetalGather,
            MetalGatherStepMemory,
            DescribeMetalGatherPipelineStatus,
            nullptr,
            ObserveMetalGatherFailure};
  case rund::kernel::NodeKind::Histogram:
    return {PrepareMetalHistogramStep, EncodeMetalHistogram,
            FinishMetalHistogram, MetalHistogramStepMemory,
            DescribeMetalHistogramPipelineStatus};
  case rund::kernel::NodeKind::Partition:
    return {PrepareMetalPartitionStep, EncodeMetalPartition,
            FinishMetalPartition, MetalPartitionStepMemory,
            DescribeMetalPartitionPipelineStatus};
  case rund::kernel::NodeKind::Reduce:
    return {PrepareMetalReduceStep, EncodeMetalReduce, FinishMetalReduce,
            MetalReduceStepMemory, DescribeMetalReducePipelineStatus};
  case rund::kernel::NodeKind::Scatter:
    return {PrepareMetalScatterStep, EncodeMetalScatter, FinishMetalScatter,
            MetalScatterStepMemory, DescribeMetalScatterPipelineStatus};
  case rund::kernel::NodeKind::ScatterReduce:
    return {PrepareMetalScatterReduceStep, EncodeMetalScatterReduce,
            FinishMetalScatterReduce, MetalScatterReduceStepMemory,
            DescribeMetalScatterReducePipelineStatus};
  case rund::kernel::NodeKind::Stencil:
    return {PrepareMetalStencilStep, EncodeMetalStencil, FinishMetalStencil};
  default:
    return MetalNumericOpsFor(kind);
  }
}

#endif

} // namespace rund::node::accel::detail
