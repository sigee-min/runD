#pragma once

#include <accel/check.hpp>

#include "adapter/api.hpp"
#include <memory>

namespace rund {
struct RuntimeStats;
}

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

using VulkanEncodedResourceFn = rund::AccelCheck (*)(
    VulkanAdapter &, const std::shared_ptr<void> &, void *);

using VulkanFinishResourceFn =
    rund::AccelCheck (*)(VulkanAdapter &, const std::shared_ptr<void> &);

[[nodiscard]] bool EnsureVulkanCommandResources(VulkanAdapter &adapter);
void DestroyVulkanCommandResources(VulkanAdapter &adapter) noexcept;

[[nodiscard]] bool BeginVulkanCommand(VulkanAdapter &adapter);
void CancelVulkanCommand(VulkanAdapter &adapter) noexcept;
void WaitForVulkanCommands(VulkanAdapter &adapter,
                           std::unique_lock<std::mutex> &lock);

void WaitForVulkanCommandSlot(VulkanAdapter &adapter,
                              std::unique_lock<std::mutex> &lock);

void EncodeVulkanComputeToComputeBarrier(VkCommandBuffer command_buffer);

[[nodiscard]] bool SubmitVulkanCommand(VulkanAdapter &adapter,
                                       bool collect_timestamp = true,
                                       rund::RuntimeStats *stats = nullptr);
[[nodiscard]] bool SubmitVulkanTransfer(VulkanAdapter &adapter,
                                        VulkanBuffer &staging,
                                        std::shared_ptr<void> target = {});
[[nodiscard]] bool SubmitVulkanCommand(VulkanAdapter &adapter,
                                       KernelCompletion completion, void *user);
[[nodiscard]] bool SubmitVulkanExternal(
    VulkanAdapter &adapter, VkCommandBuffer command, VkFence fence,
    KernelCompletion completion, void *user);

[[nodiscard]] rund::AccelCheck SubmitVulkanEncodedResources(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    VulkanEncodedResourceFn encode, VulkanFinishResourceFn finish);

#endif

} // namespace rund::node::accel::detail
