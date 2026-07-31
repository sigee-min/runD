#pragma once

#include <accel/check.hpp>

#include "model.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct CommandPlan final {
  VkCommandBufferLevel level{};
  VkCommandPoolCreateFlags pool_flags{};
  VkCommandBufferUsageFlags begin_flags{};
  VkFenceCreateFlags fence_flags{};
  bool fenced{};
  bool inherited{};
  bool valid{};
};

[[nodiscard]] constexpr CommandPlan
PlanCommand(const CommandKind kind) noexcept {
  switch (kind) {
  case CommandKind::OneShotPrimary:
    return CommandPlan{
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .pool_flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .begin_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .fenced = true,
        .valid = true,
    };
  case CommandKind::ImmutableSecondary:
    return CommandPlan{
        .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
        .inherited = true,
        .valid = true,
    };
  case CommandKind::ReusablePrimary:
    return CommandPlan{
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .fence_flags = VK_FENCE_CREATE_SIGNALED_BIT,
        .fenced = true,
        .valid = true,
    };
  }
  return {};
}

[[nodiscard]] rund::AccelCheck CreateCommand(VkDevice device,
                                             std::uint32_t queue_family,
                                             VulkanCommand &command,
                                             CommandKind kind) noexcept;

void DestroyCommand(VkDevice device, VulkanCommand &command) noexcept;

[[nodiscard]] rund::AccelCheck BeginCommand(VkDevice device,
                                            VulkanCommand &command,
                                            CommandKind kind) noexcept;

[[nodiscard]] rund::AccelCheck EndCommand(VulkanCommand &command) noexcept;

#endif

} // namespace rund::node::accel::detail
