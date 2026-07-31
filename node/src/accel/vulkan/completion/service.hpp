#pragma once

#include "../adapter/state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void RunVulkanCompletion(VulkanAdapter *adapter) noexcept;
#endif

} // namespace rund::node::accel::detail
