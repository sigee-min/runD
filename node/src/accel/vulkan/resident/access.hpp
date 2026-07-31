#pragma once

#include "../adapter/state.hpp"
#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline VulkanResidentState &
VulkanResidents(VulkanAdapter &adapter) noexcept {
  return *adapter.resident;
}

[[nodiscard]] inline const VulkanResidentState &
VulkanResidents(const VulkanAdapter &adapter) noexcept {
  return *adapter.resident;
}
#endif

} // namespace rund::node::accel::detail
