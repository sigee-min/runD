#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct BoundControl;
struct VulkanPipelineTelemetrySource;

[[nodiscard]] rund::AccelCheck PrepareVulkanMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<void> &resources,
    rund::kernel::u32 iterations = 1u);
[[nodiscard]] rund::AccelCheck
EncodeVulkanMap(VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
                void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanMap(VulkanAdapter &adapter,
                const std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck DescribeVulkanMapPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    VulkanPipelineTelemetrySource &source);

} // namespace rund::node::accel::detail
