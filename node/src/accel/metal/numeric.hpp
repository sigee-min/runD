#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../factor.hpp"
#include "../matrix.hpp"
#include "../solve.hpp"
#include "../spectrum.hpp"
#include "../transform.hpp"

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;
struct PreparedMemory;
struct MetalPipelineStatusBindings;

inline constexpr rund::kernel::u64 kMetalAlgebraLanes = 32u;

[[nodiscard]] rund::AccelCheck PrepareMetalTransform(
    const rund::AccelDevice &pick, const rund::kernel::TransformDesc &desc,
    const rund::kernel::TransformPlan &plan, const TransformBinds &bindings,
    std::shared_ptr<void> &out,
    const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
PrepareMetalMatrix(const rund::AccelDevice &pick,
                   const rund::kernel::MatrixDesc &desc,
                   const rund::kernel::MatrixPlan &plan,
                   const MatrixBinds &bindings, std::shared_ptr<void> &out,
                   const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
PrepareMetalFactor(const rund::AccelDevice &pick,
                   const rund::kernel::FactorDesc &desc,
                   const rund::kernel::FactorPlan &plan,
                   const FactorBinds &bindings, std::shared_ptr<void> &out,
                   const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
PrepareMetalSolve(const rund::AccelDevice &pick,
                  const rund::kernel::SolveDesc &desc,
                  const rund::kernel::SolvePlan &plan,
                  const SolveBinds &bindings, std::shared_ptr<void> &out,
                  const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
PrepareMetalSpectrum(const rund::AccelDevice &pick,
                     const rund::kernel::SpectrumDesc &desc,
                     const rund::kernel::SpectrumPlan &plan,
                     const SpectrumBinds &bindings, std::shared_ptr<void> &out,
                     const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalNumeric(MetalAdapter &adapter, const std::shared_ptr<void> &prepared,
                   void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalNumeric(MetalAdapter &adapter,
                   const std::shared_ptr<void> &prepared);
[[nodiscard]] PreparedMemory
MetalNumericMemory(const std::shared_ptr<void> &prepared, std::uint64_t budget);
[[nodiscard]] bool DescribeMetalNumericPipelineStatus(
    const std::shared_ptr<void> &prepared,
    MetalPipelineStatusBindings &bindings) noexcept;

[[nodiscard]] rund::AccelCheck ExecuteMetalTransform(
    const rund::AccelDevice &pick, const rund::kernel::TransformDesc &desc,
    const rund::kernel::TransformPlan &plan, const TransformBinds &bindings);

[[nodiscard]] rund::AccelCheck ExecuteMetalMatrix(
    const rund::AccelDevice &pick, const rund::kernel::MatrixDesc &desc,
    const rund::kernel::MatrixPlan &plan, const MatrixBinds &bindings);

[[nodiscard]] rund::AccelCheck ExecuteMetalFactor(
    const rund::AccelDevice &pick, const rund::kernel::FactorDesc &desc,
    const rund::kernel::FactorPlan &plan, const FactorBinds &bindings);

[[nodiscard]] rund::AccelCheck ExecuteMetalSolve(
    const rund::AccelDevice &pick, const rund::kernel::SolveDesc &desc,
    const rund::kernel::SolvePlan &plan, const SolveBinds &bindings);

[[nodiscard]] rund::AccelCheck ExecuteMetalSpectrum(
    const rund::AccelDevice &pick, const rund::kernel::SpectrumDesc &desc,
    const rund::kernel::SpectrumPlan &plan, const SpectrumBinds &bindings);

} // namespace rund::node::accel::detail
