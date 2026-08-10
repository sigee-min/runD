#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck MaterializeVulkanPrimitivePipelines(
    const rund::AccelDevice &pick, const BoundStep &step,
    const PreparedBackendManifest &manifest,
    std::shared_ptr<const VulkanKernelImmutablePipelines> &out);

[[nodiscard]] bool AppendVulkanDescriptorDependency(
    VulkanKernelProgramTemplate &program,
    VulkanKernelDescriptorDependencyKind kind, const void *identity,
    std::uint32_t descriptor_count, std::uint64_t sets_per_route,
    std::uint32_t source_node, std::uint32_t &order) noexcept;

[[nodiscard]] bool AppendVulkanMapDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanMapTemplateResources &prepared, std::uint32_t source_node,
    std::uint32_t &order) noexcept;

[[nodiscard]] bool AppendVulkanCollectiveDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanKernelImmutablePipelines &pipelines,
    const PreparedBackendManifest &manifest, std::uint32_t source_node,
    std::uint32_t &order) noexcept;

[[nodiscard]] rund::AccelCheck
FinalizeVulkanDescriptorDependencies(VulkanAdapter &adapter,
                                     VulkanKernelProgramTemplate &program,
                                     std::uint32_t *failed_node);

[[nodiscard]] rund::AccelCheck AcquireVulkanProgramTemplate(
    const rund::AccelDevice &pick, const BoundStep *steps,
    std::size_t step_count, const BackendRun *probe,
    PreparedKernelTemplateRegistry *templates, std::uint32_t *failed_node,
    VulkanKernelResources &resources);

#endif

} // namespace rund::node::accel::detail
