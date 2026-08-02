#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/segmented.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck ExecuteVulkanSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    rund::kernel::ComputeDomain domain, const SegmentedScanBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    rund::kernel::ComputeDomain domain, const SegmentedScanBinds &bindings,
    std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanSegmentedScan(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources,
                          void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanSegmentedScan(VulkanAdapter &adapter,
                          const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
