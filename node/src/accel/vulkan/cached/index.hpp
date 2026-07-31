#pragma once

#include "../adapter/pipeline.hpp"

#include <cstdint>
#include <unordered_map>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineIndex final {
  std::uint64_t descriptor_epoch = 1u;
  std::unordered_multimap<std::uint64_t, VulkanCachedPipeline *> entries{};
  std::unordered_multimap<std::uint64_t, VulkanCollectivePipeline *>
      collectives{};
};

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
