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

[[nodiscard]] rund::AccelCheck
DescribeVulkanMapPipelineCaptureDemand(const std::shared_ptr<void> &,
                                       std::uint64_t &) noexcept;
[[nodiscard]] rund::AccelCheck
DescribeVulkanSegmentedReduceCaptureDemand(const std::shared_ptr<void> &,
                                           std::uint64_t &) noexcept;
[[nodiscard]] rund::AccelCheck
DescribeVulkanSortPipelineCaptureDemand(const std::shared_ptr<void> &,
                                        std::uint64_t &) noexcept;
[[nodiscard]] rund::AccelCheck
DescribeVulkanGatherPipelineCaptureDemand(const std::shared_ptr<void> &,
                                          std::uint64_t &) noexcept;
[[nodiscard]] rund::AccelCheck
DescribeVulkanScatterReduceCaptureDemand(const std::shared_ptr<void> &,
                                         std::uint64_t &) noexcept;

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
    return VulkanKernelOps{
        .prepare = PrepareVulkanMapStep,
        .encode = EncodeVulkanMap,
        .finish = FinishVulkanMap,
        .pipeline_status = DescribeVulkanMapPipelineStatus,
        .pipeline_telemetry = DescribeVulkanMapPipelineTelemetry,
        .failure = ObserveVulkanMapFailure,
        .pipeline_capture_demand = DescribeVulkanMapPipelineCaptureDemand,
    };
  case rund::kernel::NodeKind::Scan:
    return VulkanKernelOps{
        .prepare = PrepareVulkanScanStep,
        .encode = EncodeVulkanScan,
        .finish = FinishVulkanScan,
        .pipeline_status = DescribeVulkanScanPipelineStatus,
        .pipeline_telemetry = DescribeVulkanScanPipelineTelemetry,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::SegmentedScan:
    return VulkanKernelOps{
        .prepare = PrepareVulkanSegmentedStep,
        .encode = EncodeVulkanSegmentedScan,
        .finish = FinishVulkanSegmentedScan,
        .pipeline_status = DescribeVulkanSegmentedPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::SegmentedReduce:
    return VulkanKernelOps{
        .prepare = PrepareVulkanSegmentedReduceStep,
        .encode = EncodeVulkanSegmentedReduce,
        .finish = FinishVulkanSegmentedReduce,
        .pipeline_status = DescribeVulkanSegmentedReducePipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = DescribeVulkanSegmentedReduceCaptureDemand,
    };
  case rund::kernel::NodeKind::Sort:
    return VulkanKernelOps{
        .prepare = PrepareVulkanSortStep,
        .encode = EncodeVulkanSort,
        .finish = FinishVulkanSort,
        .pipeline_status = DescribeVulkanSortPipelineStatus,
        .pipeline_telemetry = DescribeVulkanSortPipelineTelemetry,
        .failure = nullptr,
        .pipeline_capture_demand = DescribeVulkanSortPipelineCaptureDemand,
    };
  case rund::kernel::NodeKind::Compact:
    return VulkanKernelOps{
        .prepare = PrepareVulkanCompactStep,
        .encode = EncodeVulkanCompact,
        .finish = FinishVulkanCompact,
        .pipeline_status = DescribeVulkanCompactPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Gather:
    return VulkanKernelOps{
        .prepare = PrepareVulkanGatherStep,
        .encode = EncodeVulkanGather,
        .finish = FinishVulkanGather,
        .pipeline_status = DescribeVulkanGatherPipelineStatus,
        .pipeline_telemetry = DescribeVulkanGatherPipelineTelemetry,
        .failure = ObserveVulkanGatherFailure,
        .pipeline_capture_demand = DescribeVulkanGatherPipelineCaptureDemand,
    };
  case rund::kernel::NodeKind::Histogram:
    return VulkanKernelOps{
        .prepare = PrepareVulkanHistogramStep,
        .encode = EncodeVulkanHistogram,
        .finish = FinishVulkanHistogram,
        .pipeline_status = DescribeVulkanHistogramPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Partition:
    return VulkanKernelOps{
        .prepare = PrepareVulkanPartitionStep,
        .encode = EncodeVulkanPartition,
        .finish = FinishVulkanPartition,
        .pipeline_status = DescribeVulkanPartitionPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Reduce:
    return VulkanKernelOps{
        .prepare = PrepareVulkanReduceStep,
        .encode = EncodeVulkanReduce,
        .finish = FinishVulkanReduce,
        .pipeline_status = DescribeVulkanReducePipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Scatter:
    return VulkanKernelOps{
        .prepare = PrepareVulkanScatterStep,
        .encode = EncodeVulkanScatter,
        .finish = FinishVulkanScatter,
        .pipeline_status = DescribeVulkanScatterPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::ScatterReduce:
    return VulkanKernelOps{
        .prepare = PrepareVulkanScatterReduceStep,
        .encode = EncodeVulkanScatterReduce,
        .finish = FinishVulkanScatterReduce,
        .pipeline_status = DescribeVulkanScatterReducePipelineStatus,
        .pipeline_telemetry = DescribeVulkanScatterReducePipelineTelemetry,
        .failure = nullptr,
        .pipeline_capture_demand = DescribeVulkanScatterReduceCaptureDemand,
    };
  case rund::kernel::NodeKind::Stencil:
    return VulkanKernelOps{
        .prepare = PrepareVulkanStencilStep,
        .encode = EncodeVulkanStencil,
        .finish = FinishVulkanStencil,
        .pipeline_status = NoVulkanPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  default:
    return VulkanNumericOpsFor(kind);
  }
}

#endif

} // namespace rund::node::accel::detail
