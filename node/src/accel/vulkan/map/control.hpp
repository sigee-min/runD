#pragma once

#include <array>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// The push layouts are shared by the source, admission, and encode owners.
// No shader generation or command recording belongs in this compatibility
// header.
struct VulkanMapControlPush final {
  std::array<std::uint32_t, 16u> words{};
};

static_assert(sizeof(VulkanMapControlPush) == 64u);

struct VulkanGeneratedMapControlPush final {
  std::array<std::uint32_t, 32u> words{};
};

static_assert(sizeof(VulkanGeneratedMapControlPush) == 128u);

#endif

} // namespace rund::node::accel::detail
