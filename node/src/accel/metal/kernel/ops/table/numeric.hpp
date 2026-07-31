#pragma once

[[nodiscard]] inline MetalKernelOps
MetalNumericOpsFor(const rund::kernel::NodeKind kind) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return {PrepareMetalTransformStep, EncodeMetalNumeric, FinishMetalNumeric,
            MetalNumericMemory};
  case rund::kernel::NodeKind::Matrix:
    return {PrepareMetalMatrixStep, EncodeMetalNumeric, FinishMetalNumeric,
            MetalNumericMemory};
  case rund::kernel::NodeKind::Factor:
    return {PrepareMetalFactorStep, EncodeMetalNumeric, FinishMetalNumeric,
            MetalNumericMemory, DescribeMetalNumericPipelineStatus};
  case rund::kernel::NodeKind::Solve:
    return {PrepareMetalSolveStep, EncodeMetalNumeric, FinishMetalNumeric,
            MetalNumericMemory, DescribeMetalNumericPipelineStatus};
  case rund::kernel::NodeKind::Spectrum:
    return {PrepareMetalSpectrumStep, EncodeMetalNumeric, FinishMetalNumeric,
            MetalNumericMemory, DescribeMetalNumericPipelineStatus};
  default:
    return {};
  }
}
