#pragma once

#include "../local.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck
PrepareVulkanResets(VulkanAdapter &adapter, const BoundStep *steps,
                    const BoundResets *resets, KernelPreparationMode mode,
                    std::uint32_t *failed_node,
                    VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck
PrepareVulkanResetCommands(VulkanAdapter &adapter,
                           VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck PrepareVulkanDescriptorLeaseStorage(
    const BoundStep *steps, std::size_t step_count, std::uint32_t *failed_node,
    VulkanKernelResources &resources);

void DestroyPreparedVulkanKernelResources(void *raw);

void ProjectVulkanPreparedMemory(const VulkanMemoryStats before,
                                 const VulkanAdapter &adapter,
                                 VulkanKernelResources &resources,
                                 PreparedMemory &memory);

#endif

} // namespace rund::node::accel::detail
