#pragma once

#include "../adapter/state.hpp"
#include "../buffer/resident/model.hpp"

#include <kernel/program/compute/binding/model.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanCopyRange final {
  VkDeviceSize base{};
  VkDeviceSize bytes{};
  std::uint64_t offset_words{};
  std::uint64_t stride_words{};
};

[[nodiscard]] bool PlanVulkanCopyRange(
    const VulkanAdapter &adapter, const rund::kernel::ResidentBufferRef &ref,
    const VulkanBuffer *buffer, VulkanCopyRange &range) noexcept;

#endif

} // namespace rund::node::accel::detail
