#pragma once

#include "../adapter/api.hpp"

#include <array>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t kVulkanDescriptorInlineCount = 8u;

struct VulkanDescriptorScratch {
  std::array<VkDescriptorBufferInfo, kVulkanDescriptorInlineCount> inline_infos{};
  std::array<VkWriteDescriptorSet, kVulkanDescriptorInlineCount> inline_writes{};
  VkDescriptorBufferInfo* infos = nullptr;
  VkWriteDescriptorSet* writes = nullptr;
};

inline void SelectVulkanDescriptorScratch(VulkanAdapter& adapter,
                                          const std::uint32_t descriptor_count,
                                          VulkanDescriptorScratch& scratch) {
  scratch.infos = scratch.inline_infos.data();
  scratch.writes = scratch.inline_writes.data();
  if (descriptor_count > kVulkanDescriptorInlineCount) {
    adapter.descriptor_infos.resize(descriptor_count);
    adapter.descriptor_writes.resize(descriptor_count);
    scratch.infos = adapter.descriptor_infos.data();
    scratch.writes = adapter.descriptor_writes.data();
  }
}

#endif

}  // namespace rund::node::accel::detail
