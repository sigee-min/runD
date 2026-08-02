#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanReduceStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const KernelPreparationMode mode,
                        const VulkanKernelImmutablePipelines *const pipelines,
                        std::shared_ptr<void> &resources) {
  (void)mode;
  const ReduceBinds *const bindings =
      BindingsFor<ReduceBinds>(step, rund::kernel::NodeKind::Reduce);
  const auto *active = OperationFor<operation::Reduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanReduce(pick, active->desc, active->plan,
                                   step.planned->domain, *bindings, resources,
                                   pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanScatterStep(const rund::AccelDevice &pick, const BoundStep &step,
                         const KernelPreparationMode mode,
                         const VulkanKernelImmutablePipelines *const pipelines,
                         std::shared_ptr<void> &resources) {
  (void)mode;
  const ScatterBinds *const bindings =
      BindingsFor<ScatterBinds>(step, rund::kernel::NodeKind::Scatter);
  const auto *active = OperationFor<operation::Scatter>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanScatter(pick, active->desc, active->plan, *bindings,
                                    resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanStencilStep(const rund::AccelDevice &pick, const BoundStep &step,
                         const KernelPreparationMode mode,
                         const VulkanKernelImmutablePipelines *const pipelines,
                         std::shared_ptr<void> &resources) {
  (void)mode;
  const StencilBinds *const bindings =
      BindingsFor<StencilBinds>(step, rund::kernel::NodeKind::Stencil);
  const auto *active = OperationFor<operation::Stencil>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanStencil(pick, active->desc, active->plan,
                                    step.planned->domain, *bindings, resources,
                                    pipelines);
}

#include "numeric.hpp"
