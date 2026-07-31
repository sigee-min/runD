#pragma once

[[nodiscard]] rund::AccelCheck DescribeVulkanNumericPipelineStatus(
    const std::shared_ptr<void> &, VulkanPipelineStatusSource &);

[[nodiscard]] inline VulkanKernelOps
VulkanNumericOpsFor(const rund::kernel::NodeKind kind) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return {PrepareVulkanTransformStep, EncodeVulkanNumeric,
            FinishVulkanNumeric, DescribeVulkanNumericPipelineStatus};
  case rund::kernel::NodeKind::Matrix:
    return {PrepareVulkanMatrixStep, EncodeVulkanNumeric, FinishVulkanNumeric,
            DescribeVulkanNumericPipelineStatus};
  case rund::kernel::NodeKind::Factor:
    return {PrepareVulkanFactorStep, EncodeVulkanNumeric, FinishVulkanNumeric,
            DescribeVulkanNumericPipelineStatus};
  case rund::kernel::NodeKind::Solve:
    return {PrepareVulkanSolveStep, EncodeVulkanNumeric, FinishVulkanNumeric,
            DescribeVulkanNumericPipelineStatus};
  case rund::kernel::NodeKind::Spectrum:
    return {PrepareVulkanSpectrumStep, EncodeVulkanNumeric,
            FinishVulkanNumeric, DescribeVulkanNumericPipelineStatus};
  default:
    return {};
  }
}
