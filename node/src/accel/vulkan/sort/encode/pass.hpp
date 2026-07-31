#pragma once

#include <accel/check.hpp>

#include "prefix.hpp"
#include "scatter.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
EncodeVulkanSortPass(VulkanAdapter &adapter,
                     const VulkanSortEncodeResources &sort,
                     const std::size_t pass, const VkCommandBuffer command) {
  EncodeVulkanSortClassify(sort, pass, command);
  EncodeVulkanSortClassifyBarrier(sort, command);
  if (sort.block_count > 1u) {
    EncodeVulkanSortPrefix(sort, pass, command);
    EncodeVulkanSortPrefixBarrier(sort, command);
  }
  EncodeVulkanSortBase(sort, pass, command);
  EncodeVulkanSortBaseBarrier(sort, command);
  EncodeVulkanSortScatter(sort, pass, command);
  return EncodeVulkanSortTargetBarrier(adapter, sort, pass, command);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
