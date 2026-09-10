#pragma once

#include "../model.hpp"
#include "../prepare/numeric.hpp"
#include "../../../numeric.hpp"

#include <memory>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck DescribeVulkanNumericPipelineStatus(
    const std::shared_ptr<void> &, VulkanPipelineStatusSource &);

[[nodiscard]] inline VulkanKernelOps
VulkanNumericOpsFor(const rund::kernel::NodeKind kind) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return VulkanKernelOps{
        .prepare = PrepareVulkanTransformStep,
        .encode = EncodeVulkanNumeric,
        .finish = FinishVulkanNumeric,
        .pipeline_status = DescribeVulkanNumericPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Matrix:
    return VulkanKernelOps{
        .prepare = PrepareVulkanMatrixStep,
        .encode = EncodeVulkanNumeric,
        .finish = FinishVulkanNumeric,
        .pipeline_status = DescribeVulkanNumericPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Factor:
    return VulkanKernelOps{
        .prepare = PrepareVulkanFactorStep,
        .encode = EncodeVulkanNumeric,
        .finish = FinishVulkanNumeric,
        .pipeline_status = DescribeVulkanNumericPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Solve:
    return VulkanKernelOps{
        .prepare = PrepareVulkanSolveStep,
        .encode = EncodeVulkanNumeric,
        .finish = FinishVulkanNumeric,
        .pipeline_status = DescribeVulkanNumericPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  case rund::kernel::NodeKind::Spectrum:
    return VulkanKernelOps{
        .prepare = PrepareVulkanSpectrumStep,
        .encode = EncodeVulkanNumeric,
        .finish = FinishVulkanNumeric,
        .pipeline_status = DescribeVulkanNumericPipelineStatus,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  default:
    return VulkanKernelOps{
        .prepare = nullptr,
        .encode = nullptr,
        .finish = nullptr,
        .pipeline_status = nullptr,
        .pipeline_telemetry = nullptr,
        .failure = nullptr,
        .pipeline_capture_demand = nullptr,
    };
  }
}

} // namespace rund::node::accel::detail

#endif
