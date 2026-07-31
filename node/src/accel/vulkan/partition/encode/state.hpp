#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanPartitionEncodeState {
  VulkanPartitionEncodeResources *partition = nullptr;
  VkCommandBuffer command = VK_NULL_HANDLE;
  void *command_raw = nullptr;
  std::uint32_t workgroups = 0u;
};

[[nodiscard]] rund::AccelCheck LoadVulkanPartitionEncodeState(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_raw, VulkanPartitionEncodeState &state) {
  state.partition =
      static_cast<VulkanPartitionEncodeResources *>(resources.get());
  state.command = reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  state.command_raw = command_buffer_raw;
  if (state.partition == nullptr || state.partition->adapter != &adapter ||
      state.command == VK_NULL_HANDLE ||
      state.partition->classify_pipeline == nullptr ||
      state.partition->scatter_pipeline == nullptr ||
      state.partition->classify_set == VK_NULL_HANDLE ||
      state.partition->scatter_set == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  state.workgroups = static_cast<std::uint32_t>(
      (state.partition->plan.element_count + kPartitionBlockSize - 1u) /
      kPartitionBlockSize);
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
