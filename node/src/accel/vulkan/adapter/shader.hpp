#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanShader {
  std::shared_ptr<const std::vector<std::uint32_t>> words{};
  std::uint64_t hash = 0u;
};

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
