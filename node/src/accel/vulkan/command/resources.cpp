#include "resources.hpp"

#include "../command.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] constexpr const char *
FenceReason(const CommandKind kind) noexcept {
  return kind == CommandKind::OneShotPrimary
             ? "accel_vulkan_fence_failed"
             : "accel_vulkan_command_unavailable";
}

void DestroyTimestamps(VulkanAdapter &adapter) noexcept {
  for (VulkanAdapter::CommandSlot &slot : adapter.commands) {
    if (slot.timestamps != VK_NULL_HANDLE) {
      vkDestroyQueryPool(adapter.device, slot.timestamps, nullptr);
      slot.timestamps = VK_NULL_HANDLE;
    }
  }
}

void DisableTimestamps(VulkanAdapter &adapter) noexcept {
  DestroyTimestamps(adapter);
  adapter.timestamp_query_available = false;
  adapter.accel_timestamp_source = "unavailable";
}

void CreateTimestamps(VulkanAdapter &adapter) {
  if (!adapter.timestamp_query_available) {
    return;
  }
  const VkQueryPoolCreateInfo query{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2u,
  };
  for (VulkanAdapter::CommandSlot &slot : adapter.commands) {
    if (vkCreateQueryPool(adapter.device, &query, nullptr, &slot.timestamps) !=
            VK_SUCCESS ||
        slot.timestamps == VK_NULL_HANDLE) {
      DisableTimestamps(adapter);
      return;
    }
  }
}

} // namespace

rund::AccelCheck CreateCommand(const VkDevice device,
                               const std::uint32_t queue_family,
                               VulkanCommand &command,
                               const CommandKind kind) noexcept {
  if (device == VK_NULL_HANDLE || command.pool != VK_NULL_HANDLE ||
      command.buffer != VK_NULL_HANDLE || command.fence != VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const CommandPlan plan = PlanCommand(kind);
  if (!plan.valid) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const VkCommandPoolCreateInfo pool{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = plan.pool_flags,
      .queueFamilyIndex = queue_family,
  };
  if (vkCreateCommandPool(device, &pool, nullptr, &command.pool) !=
          VK_SUCCESS ||
      command.pool == VK_NULL_HANDLE) {
    DestroyCommand(device, command);
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const VkCommandBufferAllocateInfo allocation{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command.pool,
      .level = plan.level,
      .commandBufferCount = 1u,
  };
  if (vkAllocateCommandBuffers(device, &allocation, &command.buffer) !=
          VK_SUCCESS ||
      command.buffer == VK_NULL_HANDLE) {
    DestroyCommand(device, command);
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (!plan.fenced) {
    return rund::AccelCheck{true, "ok"};
  }
  const VkFenceCreateInfo fence{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = plan.fence_flags,
  };
  if (vkCreateFence(device, &fence, nullptr, &command.fence) != VK_SUCCESS ||
      command.fence == VK_NULL_HANDLE) {
    const char *const reason = FenceReason(kind);
    DestroyCommand(device, command);
    return rund::AccelCheck{false, reason};
  }
  return rund::AccelCheck{true, "ok"};
}

void DestroyCommand(const VkDevice device, VulkanCommand &command) noexcept {
  if (command.fence != VK_NULL_HANDLE) {
    vkDestroyFence(device, command.fence, nullptr);
  }
  if (command.pool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device, command.pool, nullptr);
  }
  command = {};
}

rund::AccelCheck BeginCommand(const VkDevice device, VulkanCommand &command,
                              const CommandKind kind) noexcept {
  const CommandPlan plan = PlanCommand(kind);
  if (!plan.valid || device == VK_NULL_HANDLE ||
      command.pool == VK_NULL_HANDLE || command.buffer == VK_NULL_HANDLE ||
      (plan.fenced && command.fence == VK_NULL_HANDLE) ||
      (!plan.fenced && command.fence != VK_NULL_HANDLE)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (kind == CommandKind::OneShotPrimary &&
      vkResetCommandPool(device, command.pool, 0u) != VK_SUCCESS) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  VkCommandBufferInheritanceInfo inheritance{};
  if (plan.inherited) {
    inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
  }
  const VkCommandBufferBeginInfo begin{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = plan.begin_flags,
      .pInheritanceInfo = plan.inherited ? &inheritance : nullptr,
  };
  return vkBeginCommandBuffer(command.buffer, &begin) == VK_SUCCESS
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
}

rund::AccelCheck EndCommand(VulkanCommand &command) noexcept {
  return command.buffer != VK_NULL_HANDLE &&
                 vkEndCommandBuffer(command.buffer) == VK_SUCCESS
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
}

bool EnsureVulkanCommandResources(VulkanAdapter &adapter) {
  if (adapter.commands.front().command.pool != VK_NULL_HANDLE) {
    return true;
  }
  for (VulkanAdapter::CommandSlot &slot : adapter.commands) {
    const rund::AccelCheck created =
        CreateCommand(adapter.device, adapter.compute_queue_family,
                      slot.command, CommandKind::OneShotPrimary);
    if (!created.ok) {
      SetVulkanLastError(adapter, created.reason);
      for (VulkanAdapter::CommandSlot &owned : adapter.commands) {
        DestroyCommand(adapter.device, owned.command);
      }
      return false;
    }
  }
  CreateTimestamps(adapter);
  return true;
}

void DestroyVulkanCommandResources(VulkanAdapter &adapter) noexcept {
  adapter.command_buffer = VK_NULL_HANDLE;
  adapter.recording_command = kInvalidVulkanCommand;
  DestroyTimestamps(adapter);
  for (VulkanAdapter::CommandSlot &slot : adapter.commands) {
    DestroyCommand(adapter.device, slot.command);
  }
}

#endif

} // namespace rund::node::accel::detail
