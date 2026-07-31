#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../../backend/match.hpp"
#include "../adapter/api.hpp"
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool VulkanPickOwnsAdapter(const rund::AccelDevice &pick) noexcept {
  if (!pick.check.ok || pick.api != rund::AccelApi::Vulkan ||
      pick.owner == nullptr || pick.backend.context == nullptr ||
      pick.backend.execute != ExecuteVulkan ||
      pick.owner.get() != pick.backend.context) {
    return false;
  }
  const auto *const adapter =
      static_cast<const VulkanAdapter *>(pick.backend.context);
  return SameOwner(adapter->owner_token, pick.owner);
}

VulkanAdapter *CheckedVulkanAdapter(const rund::AccelDevice &pick) noexcept {
  if (!VulkanPickOwnsAdapter(pick)) {
    return nullptr;
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == VK_NULL_HANDLE ||
      adapter->compute_queue == VK_NULL_HANDLE) {
    return nullptr;
  }
  return adapter;
}

#endif

} // namespace rund::node::accel::detail
