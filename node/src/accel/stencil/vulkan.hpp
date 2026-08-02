#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/stencil.hpp"
#include <kernel/program/compute/stencil/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck ExecuteVulkanStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const StencilBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const StencilBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanStencil(VulkanAdapter &adapter,
                    const std::shared_ptr<void> &resources,
                    void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanStencil(VulkanAdapter &adapter,
                    const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
