#pragma once

#include "../command.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanCommandLease
RecordingVulkanCommand(const VulkanAdapter &adapter) noexcept {
  return adapter.recording_command < adapter.commands.size()
             ? VulkanCommandLease{.slot = adapter.recording_command}
             : VulkanCommandLease{};
}

void ClearVulkanRecordingCommand(VulkanAdapter &adapter) noexcept {
  adapter.recording_command = kInvalidVulkanCommand;
  adapter.command_buffer = VK_NULL_HANDLE;
}

} // namespace

bool BeginVulkanCommand(VulkanAdapter &adapter) {
  if (!EnsureVulkanCommandResources(adapter) ||
      adapter.recording_command != kInvalidVulkanCommand) {
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  VulkanCommandLease command = adapter.command_ring.claim();
  if (!command) {
    ::rund::detail::counter::Accumulate(
        adapter.command_capacity_rejection_count, 1u);
    SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
    return false;
  }
  if (adapter.command_ring.active > adapter.command_inflight_peak) {
    adapter.command_inflight_peak = adapter.command_ring.active;
  }
  VulkanCommand &native = adapter.commands[command.slot].command;
  if (vkResetCommandPool(adapter.device, native.pool, 0u) != VK_SUCCESS) {
    (void)adapter.command_ring.cancel(command);
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(native.buffer, &begin) != VK_SUCCESS) {
    (void)adapter.command_ring.cancel(command);
    SetVulkanLastError(adapter, "accel_vulkan_command_unavailable");
    return false;
  }
  adapter.recording_command = command.slot;
  adapter.command_buffer = native.buffer;
  return true;
}

void CancelVulkanCommand(VulkanAdapter &adapter) noexcept {
  const VulkanCommandLease command = RecordingVulkanCommand(adapter);
  ClearVulkanRecordingCommand(adapter);
  if (command) {
    (void)adapter.command_ring.cancel(command);
    adapter.command_cv.notify_all();
  }
}

void WaitForVulkanCommands(VulkanAdapter &adapter,
                           std::unique_lock<std::mutex> &lock) {
  adapter.command_cv.wait(lock,
                          [&adapter] { return adapter.command_ring.empty(); });
}

void WaitForVulkanCommandSlot(VulkanAdapter &adapter,
                              std::unique_lock<std::mutex> &lock) {
  adapter.command_cv.wait(lock, [&adapter] {
    return adapter.command_ring.active < kVulkanCommandCapacity;
  });
}
#endif

} // namespace rund::node::accel::detail
