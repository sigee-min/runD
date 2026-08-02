#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanTransformStep(const rund::AccelDevice &pick, const BoundStep &step,
                           const KernelPreparationMode mode,
                           const VulkanKernelImmutablePipelines *const pipelines,
                           std::shared_ptr<void> &resources) {
  const TransformBinds *const bindings =
      BindingsFor<TransformBinds>(step, rund::kernel::NodeKind::Transform);
  const auto *active = OperationFor<operation::Transform>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanTransform(pick, active->desc, active->plan,
                                      *bindings, mode, resources, pipelines);
}
[[nodiscard]] inline rund::AccelCheck
PrepareVulkanMatrixStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const KernelPreparationMode mode,
                        const VulkanKernelImmutablePipelines *const pipelines,
                        std::shared_ptr<void> &resources) {
  const MatrixBinds *const bindings =
      BindingsFor<MatrixBinds>(step, rund::kernel::NodeKind::Matrix);
  const auto *active = OperationFor<operation::Matrix>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanMatrix(pick, active->desc, active->plan, *bindings,
                                   mode, resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanFactorStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const KernelPreparationMode mode,
                        const VulkanKernelImmutablePipelines *const pipelines,
                        std::shared_ptr<void> &resources) {
  const FactorBinds *const bindings =
      BindingsFor<FactorBinds>(step, rund::kernel::NodeKind::Factor);
  const auto *active = OperationFor<operation::Factor>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanFactor(pick, active->desc, active->plan, *bindings,
                                   mode, resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanSolveStep(const rund::AccelDevice &pick, const BoundStep &step,
                       const KernelPreparationMode mode,
                       const VulkanKernelImmutablePipelines *const pipelines,
                       std::shared_ptr<void> &resources) {
  const SolveBinds *const bindings =
      BindingsFor<SolveBinds>(step, rund::kernel::NodeKind::Solve);
  const auto *active = OperationFor<operation::Solve>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanSolve(pick, active->desc, active->plan, *bindings,
                                  mode, resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanSpectrumStep(const rund::AccelDevice &pick, const BoundStep &step,
                          const KernelPreparationMode mode,
                          const VulkanKernelImmutablePipelines *const pipelines,
                          std::shared_ptr<void> &resources) {
  const SpectrumBinds *const bindings =
      BindingsFor<SpectrumBinds>(step, rund::kernel::NodeKind::Spectrum);
  const auto *active = OperationFor<operation::Spectrum>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanSpectrum(pick, active->desc, active->plan,
                                     *bindings, mode, resources, pipelines);
}
