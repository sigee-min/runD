#pragma once

#include "../publish.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint64_t kVulkanPublishThreads = 256u;

[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanPublishPipeline(VulkanAdapter &);

#endif

} // namespace rund::node::accel::detail
