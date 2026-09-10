#include "internal.hpp"

#include "../../numeric/resource.hpp"
#include "../../numeric/source.hpp"

#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/stage.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanNumericStepPipeline(VulkanAdapter &adapter,
                                 const BoundStep &step) {
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Transform: {
    const auto *const active = OperationFor<operation::Transform>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash =
        rund::kernel::HashTransform(rund::kernel::TransformDesc{});
    return AcquireNumericPipeline(
        adapter, 6u, sizeof(rund::kernel::transform_stage::Batch),
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? TransformSource64() : TransformSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Matrix: {
    const auto *const active = OperationFor<operation::Matrix>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashMatrix(
        rund::kernel::MatrixDesc{.element_bytes = active->desc.element_bytes});
    const rund::kernel::ComputeDomain domain =
        active->plan.arithmetic == rund::kernel::MatrixArithmetic::Fixed
            ? rund::kernel::ComputeDomain::Fixed
            : (active->plan.arithmetic ==
                       rund::kernel::MatrixArithmetic::SignedWrap
                   ? (wide ? rund::kernel::ComputeDomain::I64
                           : rund::kernel::ComputeDomain::I32)
                   : (wide ? rund::kernel::ComputeDomain::U64
                           : rund::kernel::ComputeDomain::U32));
    return AcquireNumericPipeline(
        adapter, 4u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          domain,
                          active->plan.arithmetic ==
                                  rund::kernel::MatrixArithmetic::Fixed
                              ? active->plan.fixed_format
                              : rund::kernel::ComputeFixedFormat{}),
        wide ? MatrixSource64() : MatrixSource(),
        MatrixPolicy(active->plan.arithmetic, active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Factor: {
    const auto *const active = OperationFor<operation::Factor>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashFactor(
        rund::kernel::FactorDesc{.element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 5u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? FactorSource64() : FactorSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Solve: {
    const auto *const active = OperationFor<operation::Solve>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashSolve(
        rund::kernel::SolveDesc{.element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 6u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? SolveSource64() : SolveSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  case rund::kernel::NodeKind::Spectrum: {
    const auto *const active = OperationFor<operation::Spectrum>(step);
    if (active == nullptr) {
      return nullptr;
    }
    const bool wide = active->plan.element_bytes == sizeof(rund::kernel::u64);
    const auto hash = rund::kernel::HashSpectrum(rund::kernel::SpectrumDesc{
        .element_bytes = active->desc.element_bytes});
    return AcquireNumericPipeline(
        adapter, 5u, 0u,
        NumericPseudoPlan(hash,
                          wide ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::ComputeDomain::Fixed,
                          active->plan.fixed_format),
        wide ? SpectrumSource64() : SpectrumSource(),
        FixedPolicy(active->plan.fixed_format));
  }
  default:
    return nullptr;
  }
}

} // namespace

bool MaterializeVulkanNumericPipelines(
    VulkanAdapter &adapter, const BoundStep &step,
    VulkanKernelImmutablePipelines &pipelines) {
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Transform:
    return pipelines.append(AcquireVulkanNumericStepPipeline(adapter, step), 6u,
                            1u);
  case rund::kernel::NodeKind::Matrix:
    return pipelines.append(AcquireVulkanNumericStepPipeline(adapter, step), 4u,
                            1u);
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Spectrum:
    return pipelines.append(AcquireVulkanNumericStepPipeline(adapter, step), 5u,
                            1u);
  case rund::kernel::NodeKind::Solve:
    return pipelines.append(AcquireVulkanNumericStepPipeline(adapter, step), 6u,
                            1u);
  default:
    return false;
  }
}

#endif

} // namespace rund::node::accel::detail
