#pragma once

#include "model.hpp"

#include <accel/check.hpp>

#include <array>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

inline constexpr std::size_t VulkanTimelineWindowCapacity = 4u;

struct VulkanTimelineBatch final {
  std::span<const VkCommandBuffer> commands{};
  VulkanTimelinePoint point{};
};

// Host readiness and Device completion have different signal producers.
// Four fixed Host-ready cells preserve independent W<=4 admission without a
// higher ready value accidentally satisfying an earlier unready batch; one
// Device-done timeline preserves native queue order.
struct VulkanTimelineOwner final {
  VkDevice device{VK_NULL_HANDLE};
  std::array<VkSemaphore, VulkanTimelineWindowCapacity> ready{};
  VkSemaphore done{VK_NULL_HANDLE};
  PFN_vkSignalSemaphoreKHR signal{};
  PFN_vkWaitSemaphoresKHR wait{};
  PFN_vkGetSemaphoreCounterValueKHR counter{};
  VulkanTimelineSupport support{};
  std::mutex gate{};
  std::array<std::uint64_t, VulkanTimelineWindowCapacity> ready_signaled{};
  std::array<std::uint64_t, VulkanTimelineWindowCapacity>
      ready_wait_scheduled{};
  std::array<std::uint64_t, VulkanTimelineWindowCapacity> next_ready_value{
      1u, 1u, 1u, 1u};
  std::array<std::uint64_t, VulkanTimelineWindowCapacity>
      generation_ready_value{};
  // Snapshot taken when a generation opens. It makes an arbitrary number of
  // recurrent points cancellable before any Host signal/native submit while
  // retaining only four ready-cell counters.
  std::array<std::uint64_t, VulkanTimelineWindowCapacity>
      generation_first_ready_value{};
  std::uint64_t ordered_ready_signaled{};
  std::uint64_t done_signal_scheduled{};
  std::uint32_t next_generation{1u};
  std::uint32_t open_generation{};
  std::uint64_t next_value{1u};
  std::uint64_t generation_first_value{};
  std::uint64_t generation_last_value{};
  // Authenticated after queue acceptance; the matching wait consumes it
  // without calling the native wait primitive.
  VulkanTimelinePoint terminal_loss{};
  bool window_ready_mode{};
  VulkanTimelineInitState init_state{VulkanTimelineInitState::Uninitialized};
  rund::AccelCheck init_check{false, "compute_backend_unsupported"};
};

[[nodiscard]] VulkanTimelineSupport QueryVulkanTimelineSupport(
    VkPhysicalDevice physical_device, std::uint32_t instance_api_version,
    const std::vector<VkExtensionProperties> &extensions) noexcept;

[[nodiscard]] rund::AccelCheck
CreateVulkanTimeline(VkDevice device, VulkanTimelineSupport support,
                     VulkanTimelineOwner &owner) noexcept;

void DestroyVulkanTimeline(VulkanTimelineOwner &owner) noexcept;

[[nodiscard]] rund::AccelCheck
VulkanTimelineCapability(const VulkanTimelineOwner &owner) noexcept;

[[nodiscard]] rund::AccelCheck
ReserveVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                                std::uint32_t &generation) noexcept;

[[nodiscard]] rund::AccelCheck
ReserveVulkanTimelinePoint(VulkanTimelineOwner &owner, std::uint32_t generation,
                           VulkanTimelinePoint &point) noexcept;

// Rolls back only a generation that has not reached a Host signal or native
// queue. An accepted window is never cancellable through this pre-submit path.
[[nodiscard]] rund::AccelCheck
CancelVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                               std::uint32_t generation) noexcept;

// Read-only admission check for Host-side data publication that must not
// mutate an indirect/control buffer for a stale or out-of-order ready point.
[[nodiscard]] rund::AccelCheck
CanSignalVulkanTimelineReady(VulkanTimelineOwner &owner,
                             VulkanTimelinePoint point) noexcept;

[[nodiscard]] rund::AccelCheck
SignalVulkanTimelineReady(VulkanTimelineOwner &owner,
                          VulkanTimelinePoint point) noexcept;

[[nodiscard]] rund::AccelCheck
SubmitVulkanTimeline(VulkanAdapter &adapter,
                     std::span<const VkCommandBuffer> commands,
                     VulkanTimelinePoint point) noexcept;

// Accepts one fixed W<=4 temporal window in one Host queue call. Each submit
// record waits its exact Host-ready point, runs one retained command batch,
// and signals the matching Device-done point. This function never signals
// readiness and therefore cannot invent Host-service completion.
[[nodiscard]] rund::AccelCheck SubmitVulkanTimelineWindow(
    VulkanAdapter &adapter, std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, bool adapter_locked = false) noexcept;

// Lowers an arbitrary-Q recurrent schedule into one vkQueueSubmit call. The
// command spans may repeat only commands recorded for simultaneous use. The
// VkSubmitInfo/timeline arrays are transient O(Q) Host lowering storage; no
// Q-sized owner survives this call.
[[nodiscard]] rund::AccelCheck SubmitVulkanTimelineStream(
    VulkanAdapter &adapter, std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, bool adapter_locked = false) noexcept;

[[nodiscard]] rund::AccelCheck
WaitVulkanTimelineDone(VulkanTimelineOwner &owner, VulkanTimelinePoint point,
                       std::uint64_t timeout_ns) noexcept;

[[nodiscard]] rund::AccelCheck
CloseVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                              VulkanTimelinePoint terminal) noexcept;

[[nodiscard]] rund::AccelCheck
ReadVulkanTimeline(const VulkanTimelineOwner &owner, std::uint64_t &ready,
                   std::uint64_t &done) noexcept;

#endif

} // namespace rund::node::accel::detail
