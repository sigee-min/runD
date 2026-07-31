#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/scatter.hpp"
#include "kernel/memory.hpp"
#include "kernel/preparation.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/scatter/reduce.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct VulkanAdapter;
struct MetalPipelineStatusBindings;
struct VulkanPipelineStatusSource;
struct VulkanPipelineTelemetrySource;

[[nodiscard]] rund::AccelCheck
ExecuteMetalScatter(const rund::AccelDevice &pick,
                    const rund::kernel::ScatterDesc &desc,
                    const rund::kernel::ScatterPlan &plan,
                    const ScatterBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalScatter(const rund::AccelDevice &pick,
                                        const rund::kernel::ScatterDesc &desc,
                                        const rund::kernel::ScatterPlan &plan,
                                        const ScatterBinds &bindings,
                                        std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeMetalScatter(MetalAdapter &adapter,
                                       const std::shared_ptr<void> &resources,
                                       void *command_encoder);
[[nodiscard]] rund::AccelCheck FinishMetalScatter(MetalAdapter &adapter,
                                       const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanScatter(const rund::AccelDevice &pick,
                     const rund::kernel::ScatterDesc &desc,
                     const rund::kernel::ScatterPlan &plan,
                     const ScatterBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanScatter(
    const rund::AccelDevice &pick, const rund::kernel::ScatterDesc &desc,
    const rund::kernel::ScatterPlan &plan,
    const ScatterBinds &bindings, std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeVulkanScatter(VulkanAdapter &adapter,
                                        const std::shared_ptr<void> &resources,
                                        void *command_buffer);
[[nodiscard]] rund::AccelCheck FinishVulkanScatter(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck PrepareMetalScatterReduce(
    const rund::AccelDevice &pick,
    const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings, std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeMetalScatterReduce(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *command_encoder);
[[nodiscard]] rund::AccelCheck FinishMetalScatterReduce(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources);
[[nodiscard]] PreparedMemory MetalScatterReduceStepMemory(
    const std::shared_ptr<void> &resources, std::uint64_t budget);
[[nodiscard]] bool DescribeMetalScatterReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &bindings) noexcept;
[[nodiscard]] rund::AccelCheck PrepareVulkanScatterReduce(
    const rund::AccelDevice &pick,
    const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings, KernelPreparationMode mode,
    std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeVulkanScatterReduce(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *command_buffer);
[[nodiscard]] rund::AccelCheck FinishVulkanScatterReduce(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck DescribeVulkanScatterReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    VulkanPipelineStatusSource &source);
[[nodiscard]] rund::AccelCheck DescribeVulkanScatterReducePipelineTelemetry(
    const std::shared_ptr<void> &resources,
    VulkanPipelineTelemetrySource &source);

} // namespace rund::node::accel::detail
