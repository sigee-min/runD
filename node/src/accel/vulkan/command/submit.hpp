#pragma once

#include "../../clock.hpp"
#include "../adapter/error.hpp"
#include "../runtime/counter.hpp"
#include "../runtime/timestamp.hpp"
#include "begin.hpp"

#include <algorithm>
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

void PublishVulkanCommandLocked(
    VulkanAdapter &adapter, const VulkanCommandLease command,
    const std::uint64_t submitted_ns, const KernelCompletion completion,
    void *const user, const VulkanBuffer staging,
    std::shared_ptr<void> target, const bool timestamp, const bool timing,
    const bool force_device_lost) {
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
      .timing = timing,
      .timestamp = timestamp,
      .force_device_lost = force_device_lost,
  };
  ++adapter.pending_size;
}

[[nodiscard]] bool QueueVulkanCommand(
    VulkanAdapter &adapter, const VulkanCommandLease command,
    const std::uint64_t submitted_ns, const KernelCompletion completion,
    void *const user, const VulkanBuffer staging = {},
    std::shared_ptr<void> target = {}, const bool timestamp = true,
    const bool timing = true, const bool force_device_lost = false) {
  std::lock_guard lock{adapter.completion_mutex};
  if (adapter.completion_stop ||
      adapter.pending_size == adapter.pending.size()) {
    return false;
  }
  PublishVulkanCommandLocked(adapter, command, submitted_ns, completion, user,
                             staging, std::move(target), timestamp, timing,
                             force_device_lost);
  return true;
}

[[nodiscard]] bool SubmitVulkanCommand(VulkanAdapter &adapter,
                                       VulkanCommandLease &command,
                                       std::uint64_t &submitted_ns,
                                       const bool collect_timing = true,
                                       bool *const submitted = nullptr) {
  if (submitted != nullptr) {
    *submitted = false;
  }
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
  submitted_ns = collect_timing ? MonotonicNanoseconds() : 0u;
  const VkResult queued =
      vkQueueSubmit(adapter.compute_queue, 1u, &submit, native.fence);
  if (queued != VK_SUCCESS) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(
        adapter, VulkanFailureReason(queued, "accel_vulkan_submit_failed"));
    return false;
  }
  if (submitted != nullptr) {
    *submitted = true;
  }
  RecordVulkanCommandSubmit(adapter);
  return true;
}

} // namespace

namespace {

[[nodiscard]] bool SubmitVulkanCommandAndWait(
    VulkanAdapter &adapter, const SubmitKind kind,
    const bool collect_timestamp, rund::RuntimeStats *const stats,
    bool *const submitted) {
  if (submitted != nullptr) {
    *submitted = false;
  }
  if (stats != nullptr) {
    *stats = rund::RuntimeStats{.outcome = {.ok = true, .reason = "ok"}};
  }
  VulkanCommandLease command{};
  std::uint64_t submitted_ns = 0u;
  if (!SubmitVulkanCommand(adapter, command, submitted_ns, collect_timestamp,
                           submitted)) {
    return false;
  }
  if (stats != nullptr) {
    stats->run.work.command_submit_count = 1u;
    stats->run.work.command_capacity = kVulkanCommandCapacity;
    stats->run.work.command_inflight_peak = adapter.command_ring.active;
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
  if (adapter.device_loss_fault.take(kind)) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
    SetVulkanLastError(adapter, "compute_device_lost");
    return false;
  }
  const std::uint64_t submit_wait_ns = MonotonicNanoseconds() - submitted_ns;
  RecordVulkanCommandSubmitWaitNs(adapter, submit_wait_ns);
  if (stats != nullptr) {
    stats->run.time.command_submit_wait_ns = submit_wait_ns;
  }
  const bool timestamp =
      !collect_timestamp ||
      CollectVulkanTimestampSpan(adapter, command.slot, stats);
  (void)adapter.command_ring.cancel(command);
  adapter.command_cv.notify_all();
  return timestamp;
}

} // namespace

bool SubmitVulkanCommand(VulkanAdapter &adapter, const bool collect_timestamp,
                         rund::RuntimeStats *const stats,
                         bool *const submitted) {
  return SubmitVulkanCommandAndWait(adapter, SubmitKind::Work,
                                    collect_timestamp, stats, submitted);
}

bool SubmitVulkanTransferCommand(VulkanAdapter &adapter,
                                 bool *const submitted) {
  return SubmitVulkanCommandAndWait(adapter, SubmitKind::Transfer, false,
                                    nullptr, submitted);
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
                         const KernelCompletion completion, void *const user,
                         const bool collect_timestamp) {
  if (completion == nullptr) {
    CancelVulkanCommand(adapter);
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  std::unique_lock completion_lock{adapter.completion_mutex};
  if (adapter.completion_stop ||
      adapter.pending_size == adapter.pending.size()) {
    CancelVulkanCommand(adapter);
    SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
    return false;
  }
  VulkanCommandLease command{};
  std::uint64_t submitted_ns = 0u;
  if (!SubmitVulkanCommand(adapter, command, submitted_ns)) {
    return false;
  }
  const bool force_device_lost =
      adapter.device_loss_fault.take(SubmitKind::Work);
  PublishVulkanCommandLocked(adapter, command, submitted_ns, completion, user,
                             {}, {}, collect_timestamp, collect_timestamp,
                             force_device_lost);
  completion_lock.unlock();
  adapter.completion_cv.notify_one();
  return true;
}

bool SubmitVulkanExternal(VulkanAdapter &adapter,
                          const std::span<const VkCommandBuffer> commands,
                          const VkFence fence,
                          const KernelCompletion completion, void *const user,
                          const bool collect_timing) {
  if (commands.empty() ||
      commands.size() > std::numeric_limits<std::uint32_t>::max() ||
      std::find(commands.begin(), commands.end(), VK_NULL_HANDLE) !=
          commands.end() ||
      fence == VK_NULL_HANDLE || completion == nullptr || user == nullptr) {
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
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = static_cast<std::uint32_t>(commands.size());
  submit.pCommandBuffers = commands.data();
  const std::uint64_t submitted_ns =
      collect_timing ? MonotonicNanoseconds() : 0u;
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
      .timing = collect_timing,
      .timestamp = false,
      .external = true,
      .force_device_lost =
          adapter.device_loss_fault.take(SubmitKind::Work),
  };
  ++adapter.pending_size;
  RecordVulkanCommandSubmit(adapter);
  adapter.command_inflight_peak =
      std::max<std::uint64_t>(adapter.command_inflight_peak, 1u);
  completion_lock.unlock();
  adapter.completion_cv.notify_one();
  return true;
}

bool SubmitVulkanExternal(VulkanAdapter &adapter, const VkCommandBuffer command,
                          const VkFence fence,
                          const KernelCompletion completion, void *const user,
                          const bool collect_timing) {
  return SubmitVulkanExternal(adapter,
                              std::span<const VkCommandBuffer>{&command, 1u},
                              fence, completion, user, collect_timing);
}

#endif

} // namespace rund::node::accel::detail
