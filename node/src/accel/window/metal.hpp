#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/range.hpp"
#include "../range_aggregate/model.hpp"
#include <kernel/program/compute/window/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck
ExecuteMetalWindow(const rund::AccelDevice &pick,
                   const rund::kernel::WindowDesc &desc,
                   const rund::kernel::WindowPlan &plan,
                   const RangeBinds &bindings, const RangePlan &range);
[[nodiscard]] rund::AccelCheck PrepareMetalWindow(
    const rund::AccelDevice &pick, const rund::kernel::WindowDesc &desc,
    const rund::kernel::WindowPlan &plan, const RangeBinds &bindings,
    const RangePlan &range, std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalWindow(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
                  void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalWindow(MetalAdapter &adapter,
                  const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
