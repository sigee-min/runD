#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/graph/schema.hpp>

#include <memory>

#include "../kernel/bindings/scan.hpp"

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanBuffer;
struct VulkanStatus;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck ExecuteVulkanScan(
    const rund::AccelDevice &pick, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    const ScanBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanScan(
    const rund::AccelDevice &pick, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    const ScanBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanScan(VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
                 void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanScan(VulkanAdapter &adapter,
                 const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck ExecuteVulkanScanBuffers(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    const VulkanBuffer &input, const VulkanBuffer &output,
    bool record_dispatches, const VulkanBuffer &logical_count);
[[nodiscard]] rund::AccelCheck PrepareVulkanScanBuffers(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    const VulkanBuffer &input, const VulkanBuffer &output,
    const VulkanBuffer &logical_count, const VulkanBuffer &totals,
    VulkanStatus &status, std::shared_ptr<void> &resources,
    std::uint64_t input_offset = 0u, std::uint64_t input_range = 0u,
    std::uint64_t output_offset = 0u, std::uint64_t output_range = 0u,
    std::uint64_t logical_count_offset = 0u,
    std::uint64_t logical_count_range = 0u,
    const VulkanKernelImmutablePipelines *pipelines = nullptr,
    std::uint32_t pipeline_offset = 0u);
[[nodiscard]] rund::AccelCheck
EncodeVulkanScanBuffers(VulkanAdapter &adapter,
                        const std::shared_ptr<void> &resources,
                        void *command_buffer);
[[nodiscard]] bool VulkanScanStatusOk(const VulkanStatus &status);
[[nodiscard]] rund::kernel::u32
VulkanScanStatusFlags(const VulkanStatus &status);

} // namespace rund::node::accel::detail
