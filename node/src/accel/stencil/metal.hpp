#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/range.hpp"
#include "../range_aggregate/model.hpp"
#include <kernel/program/compute/stencil/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck ExecuteMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const RangeBinds &bindings, const RangePlan &range);
[[nodiscard]] rund::AccelCheck PrepareMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const RangeBinds &bindings, const RangePlan &range,
    std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalStencil(MetalAdapter &adapter,
                   const std::shared_ptr<void> &resources,
                   void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalStencil(MetalAdapter &adapter,
                   const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
