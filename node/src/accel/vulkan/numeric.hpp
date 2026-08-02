#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../factor.hpp"
#include "../matrix.hpp"
#include "../solve.hpp"
#include "../spectrum.hpp"
#include "../transform.hpp"
#include "../kernel/preparation.hpp"

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck PrepareVulkanTransform(
    const rund::AccelDevice &pick, const rund::kernel::TransformDesc &desc,
    const rund::kernel::TransformPlan &plan,
    const TransformBinds &bindings, KernelPreparationMode mode,
    std::shared_ptr<void> &out,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck PrepareVulkanMatrix(
    const rund::AccelDevice &pick, const rund::kernel::MatrixDesc &desc,
    const rund::kernel::MatrixPlan &plan,
    const MatrixBinds &bindings, KernelPreparationMode mode,
    std::shared_ptr<void> &out,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck PrepareVulkanFactor(
    const rund::AccelDevice &pick, const rund::kernel::FactorDesc &desc,
    const rund::kernel::FactorPlan &plan,
    const FactorBinds &bindings, KernelPreparationMode mode,
    std::shared_ptr<void> &out,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck PrepareVulkanSolve(
    const rund::AccelDevice &pick, const rund::kernel::SolveDesc &desc,
    const rund::kernel::SolvePlan &plan, const SolveBinds &bindings,
    KernelPreparationMode mode, std::shared_ptr<void> &out,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck PrepareVulkanSpectrum(
    const rund::AccelDevice &pick, const rund::kernel::SpectrumDesc &desc,
    const rund::kernel::SpectrumPlan &plan,
    const SpectrumBinds &bindings, KernelPreparationMode mode,
    std::shared_ptr<void> &out,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanNumeric(VulkanAdapter &adapter,
                    const std::shared_ptr<void> &prepared,
                    void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanNumeric(VulkanAdapter &adapter,
                    const std::shared_ptr<void> &prepared);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanTransform(const rund::AccelDevice &pick,
                       const rund::kernel::TransformDesc &desc,
                       const rund::kernel::TransformPlan &plan,
                       const TransformBinds &bindings);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanMatrix(const rund::AccelDevice &pick,
                    const rund::kernel::MatrixDesc &desc,
                    const rund::kernel::MatrixPlan &plan,
                    const MatrixBinds &bindings);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanFactor(const rund::AccelDevice &pick,
                    const rund::kernel::FactorDesc &desc,
                    const rund::kernel::FactorPlan &plan,
                    const FactorBinds &bindings);

[[nodiscard]] rund::AccelCheck ExecuteVulkanSolve(
    const rund::AccelDevice &pick, const rund::kernel::SolveDesc &desc,
    const rund::kernel::SolvePlan &plan, const SolveBinds &bindings);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanSpectrum(const rund::AccelDevice &pick,
                      const rund::kernel::SpectrumDesc &desc,
                      const rund::kernel::SpectrumPlan &plan,
                      const SpectrumBinds &bindings);

} // namespace rund::node::accel::detail
