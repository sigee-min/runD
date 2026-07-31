#pragma once

#include "../../../descriptor.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint32_t kInlineSetLayoutCount = 8u;

struct SetLayoutScratch {
  std::array<VkDescriptorSetLayout, kInlineSetLayoutCount> inlines{};
  std::vector<VkDescriptorSetLayout> heap{};
  VkDescriptorSetLayout* data = nullptr;
};

void FillSetLayouts(SetLayoutScratch& scratch,
                    const VkDescriptorSetLayout layout,
                    const std::uint32_t count) {
  if (count > kInlineSetLayoutCount) {
    scratch.heap.assign(count, layout);
    scratch.data = scratch.heap.data();
    return;
  }
  scratch.inlines.fill(layout);
  scratch.data = scratch.inlines.data();
}

#endif

}  // namespace rund::node::accel::detail
