#pragma once

#include "../../clock.hpp"
#include "../runtime/counter.hpp"
#include "../runtime/timestamp.hpp"
#include "begin.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool EndVulkanCommand(VulkanAdapter &adapter,
                                    VulkanCommandLease &command,
                                    const char *&reason) {
  command = RecordingVulkanCommand(adapter);
  if (!command) {
    reason = "accel_vulkan_command_unavailable";
    return false;
  }
  VulkanCommand &native = adapter.commands[command.slot].command;
  const VkResult ended = vkEndCommandBuffer(native.buffer);
  const VkResult reset = ended == VK_SUCCESS
                             ? vkResetFences(adapter.device, 1u, &native.fence)
                             : VK_SUCCESS;
  if (ended != VK_SUCCESS || reset != VK_SUCCESS) {
    CancelVulkanCommand(adapter);
    reason = VulkanFailureReason(ended != VK_SUCCESS ? ended : reset,
                                 "accel_vulkan_command_unavailable");
    return false;
  }
  if (!adapter.command_ring.publish(command)) {
    CancelVulkanCommand(adapter);
    reason = "accel_vulkan_command_sequence";
    return false;
  }
  ClearVulkanRecordingCommand(adapter);
  return true;
}

[[nodiscard]] bool QueueVulkanCommand(
    VulkanAdapter &adapter, const VulkanCommandLease command,
    const std::uint64_t submitted_ns, const KernelCompletion completion,
    void *const user, const VulkanBuffer staging = {},
    std::shared_ptr<void> target = {}, const bool timestamp = true,
    const bool force_device_lost = false) {
  std::lock_guard lock{adapter.completion_mutex};
  if (adapter.completion_stop ||
      adapter.pending_size == adapter.pending.size()) {
    return false;
  }
  const std::size_t tail =
      (adapter.pending_head + adapter.pending_size) % adapter.pending.size();
  adapter.pending[tail] = VulkanAdapter::Pending{
      .command = command,
      .completion = completion,
      .user = user,
      .submitted_ns = submitted_ns,
      .staging = staging,
      .target = std::move(target),
      .inflight = adapter.command_ring.active,
      .timestamp = timestamp,
      .force_device_lost = force_device_lost,
  };
  ++adapter.pending_size;
  return true;
}

[[nodiscard]] bool SubmitVulkanCommand(VulkanAdapter &adapter,
                                       VulkanCommandLease &command,
                                       std::uint64_t &submitted_ns) {
  const char *reason = "ok";
  if (!EndVulkanCommand(adapter, command, reason)) {
    SetVulkanLastError(adapter, reason);
    return false;
  }
  VulkanCommand &native = adapter.commands[command.slot].command;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1u;
  submit.pCommandBuffers = &native.buffer;
  submitted_ns = MonotonicNanoseconds();
  const VkResult submitted =
      vkQueueSubmit(adapter.compute_queue, 1u, &submit, native.fence);
  if (submitted != VK_SUCCESS) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(
        adapter, VulkanFailureReason(submitted, "accel_vulkan_submit_failed"));
    return false;
  }
  return true;
}

} // namespace

bool SubmitVulkanCommand(VulkanAdapter &adapter, const bool collect_timestamp,
                         rund::RuntimeStats *const stats) {
  if (stats != nullptr) {
    *stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  }
  VulkanCommandLease command{};
  std::uint64_t submitted_ns = 0u;
  if (!SubmitVulkanCommand(adapter, command, submitted_ns)) {
    return false;
  }
  if (stats != nullptr) {
    stats->command_submit_count = 1u;
    stats->command_capacity = kVulkanCommandCapacity;
    stats->command_inflight_peak = adapter.command_ring.active;
  }
  VulkanCommand &native = adapter.commands[command.slot].command;
  const VkResult waited =
      vkWaitForFences(adapter.device, 1u, &native.fence, VK_TRUE,
                      std::numeric_limits<std::uint64_t>::max());
  if (waited != VK_SUCCESS) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(
        adapter, VulkanFailureReason(waited, "accel_vulkan_fence_failed"));
    return false;
  }
  if (adapter.fault_device_lost_once.exchange(false,
                                              std::memory_order_relaxed)) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(adapter, "compute_device_lost");
    return false;
  }
  const std::uint64_t submit_wait_ns = MonotonicNanoseconds() - submitted_ns;
  RecordVulkanCommandSubmitWaitNs(adapter, submit_wait_ns);
  if (stats != nullptr) {
    stats->command_submit_wait_ns = submit_wait_ns;
  }
  const bool timestamp =
      !collect_timestamp ||
      CollectVulkanTimestampSpan(adapter, command.slot, stats);
  (void)adapter.command_ring.cancel(command);
  adapter.command_cv.notify_all();
  return timestamp;
}

bool SubmitVulkanTransfer(VulkanAdapter &adapter, VulkanBuffer &staging,
                          std::shared_ptr<void> target) {
  if (staging.buffer == VK_NULL_HANDLE || staging.memory == VK_NULL_HANDLE) {
    CancelVulkanCommand(adapter);
    SetVulkanLastError(adapter, "accel_vulkan_transfer_invalid");
    return false;
  }
  {
    std::lock_guard lock{adapter.completion_mutex};
    if (adapter.completion_stop ||
        adapter.pending_size == adapter.pending.size()) {
      CancelVulkanCommand(adapter);
      SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
      return false;
    }
  }
  VulkanCommandLease command{};
  std::uint64_t submitted_ns = 0u;
  if (!SubmitVulkanCommand(adapter, command, submitted_ns)) {
    return false;
  }
  if (!QueueVulkanCommand(adapter, command, submitted_ns, nullptr, nullptr,
                          staging, std::move(target), false)) {
    (void)vkWaitForFences(adapter.device, 1u,
                          &adapter.commands[command.slot].command.fence,
                          VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  staging = {};
  adapter.completion_cv.notify_one();
  return true;
}

bool SubmitVulkanCommand(VulkanAdapter &adapter,
                         const KernelCompletion completion, void *const user) {
  if (completion == nullptr) {
    CancelVulkanCommand(adapter);
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  {
    std::lock_guard lock{adapter.completion_mutex};
    if (adapter.completion_stop ||
        adapter.pending_size == adapter.pending.size()) {
      CancelVulkanCommand(adapter);
      SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
      return false;
    }
  }
  VulkanCommandLease command{};
  std::uint64_t submitted_ns = 0u;
  if (!SubmitVulkanCommand(adapter, command, submitted_ns)) {
    return false;
  }
  const bool force_device_lost =
      adapter.fault_device_lost_once.exchange(false, std::memory_order_relaxed);
  if (!QueueVulkanCommand(adapter, command, submitted_ns, completion, user, {},
                          {}, true, force_device_lost)) {
    // Capacity is pre-proved while the adapter mutex serializes publishers;
    // reaching this branch means service shutdown violated the owner lifetime.
    (void)vkWaitForFences(adapter.device, 1u,
                          &adapter.commands[command.slot].command.fence,
                          VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  adapter.completion_cv.notify_one();
  return true;
}

bool SubmitVulkanExternal(VulkanAdapter &adapter, const VkCommandBuffer command,
                          const VkFence fence,
                          const KernelCompletion completion, void *const user) {
  if (command == VK_NULL_HANDLE || fence == VK_NULL_HANDLE ||
      completion == nullptr || user == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  std::unique_lock completion_lock{adapter.completion_mutex};
  if (adapter.completion_stop ||
      adapter.pending_size == adapter.pending.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
    return false;
  }
  const VkResult reset = vkResetFences(adapter.device, 1u, &fence);
  if (reset != VK_SUCCESS) {
    SetVulkanLastError(adapter, VulkanFailureReason(
                                    reset, "accel_vulkan_command_unavailable"));
    return false;
  }
  const VkSubmitInfo submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1u,
      .pCommandBuffers = &command,
  };
  const std::uint64_t submitted_ns = MonotonicNanoseconds();
  const VkResult submitted =
      vkQueueSubmit(adapter.compute_queue, 1u, &submit, fence);
  if (submitted != VK_SUCCESS) {
    SetVulkanLastError(
        adapter, VulkanFailureReason(submitted, "accel_vulkan_submit_failed"));
    return false;
  }
  const std::size_t tail =
      (adapter.pending_head + adapter.pending_size) % adapter.pending.size();
  adapter.pending[tail] = VulkanAdapter::Pending{
      .external_fence = fence,
      .completion = completion,
      .user = user,
      .submitted_ns = submitted_ns,
      .inflight = 1u,
      .timestamp = false,
      .external = true,
      .force_device_lost = adapter.fault_device_lost_once.exchange(
          false, std::memory_order_relaxed),
  };
  ++adapter.pending_size;
  ++adapter.command_submit_count;
  adapter.command_inflight_peak =
      std::max<std::uint64_t>(adapter.command_inflight_peak, 1u);
  completion_lock.unlock();
  adapter.completion_cv.notify_one();
  return true;
}

#endif

} // namespace rund::node::accel::detail
