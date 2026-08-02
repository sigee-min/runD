#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/compact.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/compact/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;
struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck
ExecuteMetalCompact(const rund::AccelDevice &pick,
                    const rund::kernel::CompactDesc &desc,
                    const rund::kernel::CompactPlan &plan,
                    const CompactBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalCompact(const rund::AccelDevice &pick,
                                        const rund::kernel::CompactDesc &desc,
                                        const rund::kernel::CompactPlan &plan,
                                        const CompactBinds &bindings,
                                        std::shared_ptr<void> &resources,
                                        const MetalKernelImmutablePipelines *pipelines =
                                            nullptr);
[[nodiscard]] rund::AccelCheck EncodeMetalCompact(MetalAdapter &adapter,
                                       const std::shared_ptr<void> &resources,
                                       void *command_encoder);
[[nodiscard]] rund::AccelCheck FinishMetalCompact(MetalAdapter &adapter,
                                       const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanCompact(const rund::AccelDevice &pick,
                     const rund::kernel::CompactDesc &desc,
                     const rund::kernel::CompactPlan &plan,
                     const CompactBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanCompact(
    const rund::AccelDevice &pick, const rund::kernel::CompactDesc &desc,
    const rund::kernel::CompactPlan &plan,
    const CompactBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck EncodeVulkanCompact(VulkanAdapter &adapter,
                                        const std::shared_ptr<void> &resources,
                                        void *command_buffer);
[[nodiscard]] rund::AccelCheck FinishVulkanCompact(VulkanAdapter &adapter,
                                        const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
