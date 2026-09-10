#pragma once

#include "../owner.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace timeline_owner {

[[nodiscard]] rund::AccelCheck unsupported() noexcept;
[[nodiscard]] rund::AccelCheck invalid() noexcept;
[[nodiscard]] rund::AccelCheck failed() noexcept;

void record_failure(VulkanTimelineOwner &owner,
                    rund::AccelCheck check) noexcept;

[[nodiscard]] PFN_vkVoidFunction load(VkDevice device,
                                      VulkanTimelineRoute route,
                                      const char *core,
                                      const char *extension) noexcept;

[[nodiscard]] bool valid(const VulkanTimelineOwner &owner) noexcept;

[[nodiscard]] bool within_difference(const VulkanTimelineOwner &owner,
                                     VkSemaphore semaphore,
                                     std::uint64_t value,
                                     bool strictly_future) noexcept;

[[nodiscard]] bool valid_ready_signal(const VulkanTimelineOwner &owner,
                                      VulkanTimelinePoint point) noexcept;

[[nodiscard]] rund::AccelCheck SubmitVulkanTimelineBatches(
    VulkanAdapter &adapter, std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, bool adapter_locked) noexcept;

} // namespace timeline_owner

#endif

} // namespace rund::node::accel::detail
