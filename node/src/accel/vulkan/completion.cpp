#include <accel/check.hpp>

#include "../clock.hpp"
#include "local.hpp"

#include "completion/service.hpp"
#include "runtime/counter.hpp"
#include "runtime/timestamp.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void RunVulkanCompletion(VulkanAdapter *const adapter) noexcept {
  for (;;) {
    VulkanAdapter::Pending pending{};
    {
      std::unique_lock lock{adapter->completion_mutex};
      adapter->completion_cv.wait(lock, [&] {
        return adapter->pending_size != 0u || adapter->completion_stop;
      });
      if (adapter->pending_size == 0u && adapter->completion_stop) {
        return;
      }
      pending = std::move(adapter->pending[adapter->pending_head]);
      adapter->pending[adapter->pending_head] = {};
      adapter->pending_head =
          (adapter->pending_head + 1u) % adapter->pending.size();
      --adapter->pending_size;
    }

    KernelResult result{
        .check = rund::AccelCheck{true, "ok"},
        .stats = {.command_submit_count = 1u,
                  .command_capacity = kVulkanCommandCapacity,
                  .command_inflight_peak = pending.inflight,
                  .ok = true,
                  .reason = "ok"},
    };
    const VkFence fence =
        pending.external
            ? pending.external_fence
            : (!pending.command ||
                       pending.command.slot >= adapter->commands.size()
                   ? VK_NULL_HANDLE
                   : adapter->commands[pending.command.slot].command.fence);
    const VkResult waited =
        fence == VK_NULL_HANDLE
            ? VK_ERROR_UNKNOWN
            : vkWaitForFences(adapter->device, 1u, &fence, VK_TRUE,
                              std::numeric_limits<std::uint64_t>::max());
    if (waited != VK_SUCCESS) {
      result.check = rund::AccelCheck{
          false, VulkanFailureReason(waited, "accel_vulkan_fence_failed")};
    }
    if (pending.force_device_lost) {
      result.check = rund::AccelCheck{false, "compute_device_lost"};
    }

    {
      std::lock_guard lock{adapter->mutex};
      result.stats.command_submit_wait_ns =
          MonotonicNanoseconds() - pending.submitted_ns;
      RecordVulkanCommandSubmitWaitNs(*adapter,
                                      result.stats.command_submit_wait_ns);
      if (result.check.ok && !pending.external && pending.timestamp &&
          !CollectVulkanTimestampSpan(*adapter, pending.command.slot,
                                      &result.stats)) {
        result.check = rund::AccelCheck{false, VulkanLastError(adapter)};
      }
      if (!pending.external &&
          !adapter->command_ring.retirable(pending.command)) {
        result.check = rund::AccelCheck{false, "accel_vulkan_command_sequence"};
      }
      if (!result.check.ok) {
        SetVulkanLastError(*adapter, result.check.reason);
      }
      if (!pending.external) {
        ReleaseVulkanBuffer(*adapter, pending.staging);
      }
    }
    if (pending.completion != nullptr) {
      pending.completion(pending.user, result);
    }
    if (!pending.external) {
      // The fence has completed, so release transfer-retained resident storage
      // before publishing retirement to synchronous observers.
      pending.target.reset();
      std::lock_guard lock{adapter->mutex};
      if (!adapter->command_ring.retire(pending.command)) {
        SetVulkanLastError(*adapter, "accel_vulkan_command_sequence");
      }
      adapter->command_cv.notify_all();
    }
  }
}

#endif

} // namespace rund::node::accel::detail
