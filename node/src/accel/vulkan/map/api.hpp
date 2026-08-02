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
struct VulkanMapDescriptorArena;
struct VulkanMapTemplateResources;
struct VulkanPipelineTelemetrySource;

// Exact immutable specialization identity used by proved Pipeline
// recurrence. Route buffer handles are intentionally absent; only the plan
// and source-relevant binding layout participate.
[[nodiscard]] bool VulkanMapTemplateMatches(
    const VulkanMapTemplateResources &prepared, const VulkanAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept;

[[nodiscard]] rund::AccelCheck PrepareVulkanMapTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared);

// Consumes the one-shot recurrence artifact after common lowering reserved
// its final backend source envelope. Stride edits happen in place and that
// same string allocation moves into the adapter cache.
[[nodiscard]] rund::AccelCheck PrepareVulkanMapOwnedTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact &&artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared);

[[nodiscard]] rund::AccelCheck PrepareVulkanMapRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources,
    rund::kernel::u32 iterations = 1u);

// Route-only materialization for a recurrence already admitted by common and
// compiled into `prepared`. No source artifact is accepted, so equal groups
// can only lease descriptors from the registry-owned immutable template.
[[nodiscard]] rund::AccelCheck PrepareVulkanMapProvedRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control, bool history_recurrence,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources,
    rund::kernel::u32 iterations = 1u);

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
