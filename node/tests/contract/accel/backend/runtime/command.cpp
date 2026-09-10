#include "local.hpp"

#include "src/accel/vulkan/command/resources.hpp"
#include "src/accel/vulkan/command/ring.hpp"

#include <accel/runtime.hpp>

#include <array>
#include <string_view>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

consteval bool VulkanCommandRingContract() {
  detail::VulkanCommandRing ring{};
  std::array<detail::VulkanCommandLease, detail::kVulkanCommandCapacity>
      commands{};
  for (auto &command : commands) {
    command = ring.claim();
    if (!command || !ring.publish(command)) {
      return false;
    }
  }
  if (ring.claim() || ring.retire(commands[1])) {
    return false;
  }
  for (const auto command : commands) {
    if (!ring.retire(command)) {
      return false;
    }
  }
  detail::VulkanCommandLease terminal = ring.claim();
  ring.next_sequence = kCounterMaximum;
  return ring.empty() == false && !ring.publish(terminal) &&
         ring.cancel(terminal) && ring.empty();
}

static_assert(VulkanCommandRingContract());

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
consteval bool VulkanCommandResourceContract() {
  constexpr detail::CommandPlan primary =
      detail::PlanCommand(detail::CommandKind::OneShotPrimary);
  constexpr detail::CommandPlan secondary =
      detail::PlanCommand(detail::CommandKind::ImmutableSecondary);
  constexpr detail::CommandPlan reusable =
      detail::PlanCommand(detail::CommandKind::ReusablePrimary);
  constexpr detail::CommandPlan invalid =
      detail::PlanCommand(static_cast<detail::CommandKind>(0xffu));
  return primary.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
         primary.pool_flags ==
             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT &&
         primary.begin_flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT &&
         primary.fence_flags == 0u && primary.fenced && !primary.inherited &&
         primary.valid &&
         secondary.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
         secondary.pool_flags == 0u && secondary.begin_flags == 0u &&
         secondary.fence_flags == 0u && !secondary.fenced &&
         secondary.inherited && secondary.valid &&
         reusable.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
         reusable.pool_flags == 0u &&
         reusable.begin_flags == VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT &&
         reusable.fence_flags == VK_FENCE_CREATE_SIGNALED_BIT &&
         reusable.fenced && !reusable.inherited && reusable.valid &&
         !invalid.valid;
}

static_assert(VulkanCommandResourceContract());
#endif

} // namespace

bool CheckVulkanCommandFailure() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  detail::VulkanCommand command{};
  const rund::AccelCheck rejected = detail::CreateCommand(
      VK_NULL_HANDLE, 0u, command, detail::CommandKind::ReusablePrimary);
  return !rejected.ok &&
         rejected.reason ==
             std::string_view{"accel_vulkan_command_unavailable"} &&
         command.pool == VK_NULL_HANDLE && command.buffer == VK_NULL_HANDLE &&
         command.fence == VK_NULL_HANDLE;
#else
  return true;
#endif
}

} // namespace node_accel_contract::backend_runtime
