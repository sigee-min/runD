#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/reduce.hpp"

#include <kernel/program/compute/reduce/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck ExecuteVulkanReduce(
    const rund::AccelDevice &pick, const rund::kernel::ReduceDesc &desc,
    const rund::kernel::ReducePlan &plan, rund::kernel::ComputeDomain domain,
    const ReduceBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanReduce(
    const rund::AccelDevice &pick, const rund::kernel::ReduceDesc &desc,
    const rund::kernel::ReducePlan &plan, rund::kernel::ComputeDomain domain,
    const ReduceBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanReduce(VulkanAdapter &adapter,
                   const std::shared_ptr<void> &resources,
                   void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanReduce(VulkanAdapter &adapter,
                   const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
