#pragma once

#include "create.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK) &&                                      \
    defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
[[nodiscard]] inline std::shared_ptr<VulkanAdapter>
VulkanAdapterFromCreatedDevice(const VkInstance instance,
                               const VkPhysicalDevice physical_device,
                               const VulkanCreatedDevice &created) {
  auto adapter = std::make_shared<VulkanAdapter>();
  adapter->instance = instance;
  adapter->physical_device = physical_device;
  vkGetPhysicalDeviceMemoryProperties(physical_device,
                                      &adapter->memory_properties);
  adapter->device = created.device;
  adapter->compute_queue = created.queue;
  adapter->compute_queue_family = created.queue_family;
  adapter->timestamp_valid_bits = created.timestamp_valid_bits;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  adapter->caps = rund::kernel::ComputeCaps{
      .api = rund::kernel::ComputeApi::Vulkan,
      .device_bytes = VulkanDeviceMemoryBytes(
          physical_device, adapter->memory_properties, created.extensions),
      .staging_bytes = kVulkanOneMiB,
      .max_window_tiles =
          rund::kernel::compute_lowering_detail::VulkanMapTileCapacity(
              properties.limits.maxComputeWorkGroupCount[0]),
      .subgroup_width = 1u,
      .ok = true,
      .reason = "ok",
  };
  adapter->glslang_validator_path = RUND_NODE_GLSLANG_VALIDATOR_PATH;
#if defined(RUND_NODE_HAVE_SPIRV_VAL)
  adapter->spirv_val_path = RUND_NODE_SPIRV_VAL_PATH;
#else
  adapter->spirv_val_path.clear();
#endif
  PopulateVulkanDriverProperties(
      instance, properties,
      VulkanDeviceSupportsDriverProperties(physical_device, created.extensions),
      *adapter);
  adapter->caps.storage_alignment =
      std::max<std::uint64_t>(sizeof(std::uint32_t), adapter->storage_align);
  TryCreateVulkanTimestampQueryPool(*adapter);
  if (!StartVulkanCompletionService(*adapter)) {
    // The discovery owner destroys the instance on a rejected pick; keep the
    // adapter responsible only for the logical device in this branch.
    adapter->instance = VK_NULL_HANDLE;
    return {};
  }
  return adapter;
}
#endif

} // namespace rund::node::accel::detail
