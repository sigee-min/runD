#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/range.hpp"
#include "../kernel/preparation.hpp"
#include "../range_aggregate/model.hpp"
#include <kernel/program/compute/window/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;
struct BoundControl;

[[nodiscard]] rund::AccelCheck
ExecuteVulkanWindow(const rund::AccelDevice &pick,
                    const rund::kernel::WindowDesc &desc,
                    const rund::kernel::WindowPlan &plan,
                    const RangeBinds &bindings, const RangePlan &range);
[[nodiscard]] rund::AccelCheck PrepareVulkanWindow(
    const rund::AccelDevice &pick, const rund::kernel::WindowDesc &desc,
    const rund::kernel::WindowPlan &plan, const RangeBinds &bindings,
    const RangePlan &range, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr,
    const BoundControl *control = nullptr,
    KernelPreparationMode mode = KernelPreparationMode::Standalone);
[[nodiscard]] rund::AccelCheck
EncodeVulkanWindow(VulkanAdapter &adapter,
                   const std::shared_ptr<void> &resources,
                   void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanWindow(VulkanAdapter &adapter,
                   const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
