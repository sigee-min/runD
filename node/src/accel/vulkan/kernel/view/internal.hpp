#pragma once

#include "../view.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t kVulkanViewDescriptorCount = 2u;
inline constexpr std::uint32_t kVulkanViewBlockSize = 256u;

struct VulkanViewParams final {
  std::uint64_t count{};
  std::uint64_t source_offset_words{};
  std::uint64_t source_stride_words{};
  std::uint64_t target_offset_words{};
  std::uint64_t target_stride_words{};
  std::uint32_t element_words{};
  std::uint32_t reserved{};
};

static_assert(sizeof(VulkanViewParams) == 48u);

#endif

} // namespace rund::node::accel::detail
