#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

[[nodiscard]] inline rund::AccelCheck
PrepareMetalTransformStep(const rund::AccelDevice &pick, const BoundStep &step,
                          std::shared_ptr<void> &resources) {
  const TransformBinds *const bindings =
      BindingsFor<TransformBinds>(step, rund::kernel::NodeKind::Transform);
  const auto *active = OperationFor<operation::Transform>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalTransform(pick, active->desc, active->plan,
                                     *bindings, resources);
}
[[nodiscard]] inline rund::AccelCheck
PrepareMetalMatrixStep(const rund::AccelDevice &pick, const BoundStep &step,
                       std::shared_ptr<void> &resources) {
  const MatrixBinds *const bindings =
      BindingsFor<MatrixBinds>(step, rund::kernel::NodeKind::Matrix);
  const auto *active = OperationFor<operation::Matrix>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalMatrix(pick, active->desc, active->plan, *bindings,
                                  resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalFactorStep(const rund::AccelDevice &pick, const BoundStep &step,
                       std::shared_ptr<void> &resources) {
  const FactorBinds *const bindings =
      BindingsFor<FactorBinds>(step, rund::kernel::NodeKind::Factor);
  const auto *active = OperationFor<operation::Factor>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalFactor(pick, active->desc, active->plan, *bindings,
                                  resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalSolveStep(const rund::AccelDevice &pick, const BoundStep &step,
                      std::shared_ptr<void> &resources) {
  const SolveBinds *const bindings =
      BindingsFor<SolveBinds>(step, rund::kernel::NodeKind::Solve);
  const auto *active = OperationFor<operation::Solve>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalSolve(pick, active->desc, active->plan, *bindings,
                                 resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalSpectrumStep(const rund::AccelDevice &pick, const BoundStep &step,
                         std::shared_ptr<void> &resources) {
  const SpectrumBinds *const bindings =
      BindingsFor<SpectrumBinds>(step, rund::kernel::NodeKind::Spectrum);
  const auto *active = OperationFor<operation::Spectrum>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalSpectrum(pick, active->desc, active->plan, *bindings,
                                    resources);
}
