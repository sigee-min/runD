#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/sort.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/sort/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;
struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;
[[nodiscard]] rund::AccelCheck
ExecuteMetalSort(const rund::AccelDevice &pick,
                 const rund::kernel::SortDesc &desc,
                 const rund::kernel::SortPlan &plan,
                 rund::kernel::ComputeDomain domain, const SortBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalSort(
    const rund::AccelDevice &pick, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, rund::kernel::ComputeDomain domain,
    const SortBinds &bindings, std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalSort(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
                void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalSort(MetalAdapter &adapter, const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck ExecuteVulkanSort(
    const rund::AccelDevice &pick, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, rund::kernel::ComputeDomain domain,
    const SortBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanSort(
    const rund::AccelDevice &pick, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, rund::kernel::ComputeDomain domain,
    const SortBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanSort(VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
                 void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanSort(VulkanAdapter &adapter,
                 const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
