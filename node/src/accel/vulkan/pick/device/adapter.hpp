#pragma once

#include "create.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK) &&                                      \
    defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
[[nodiscard]] inline std::shared_ptr<VulkanAdapter>
VulkanAdapterFromCreatedDevice(
    const VkInstance instance, const VkPhysicalDevice physical_device,
    const VulkanCreatedDevice &created,
    const std::uint64_t persistent_stream_submit_capacity) {
  // A completion callback is permitted to release the final public Device.
  // Deleting the adapter inline on either native worker would make its
  // destructor join the current thread. Route only that final cold deletion
  // to a neutral thread; ordinary lifetime and all warm work remain direct.
  const auto destroy = [](VulkanAdapter *const adapter) noexcept {
    if (adapter == nullptr) {
      return;
    }
    const std::thread::id self = std::this_thread::get_id();
    if ((adapter->completion_thread.joinable() &&
         adapter->completion_thread.get_id() == self) ||
        (adapter->residency_thread.joinable() &&
         adapter->residency_thread.get_id() == self)) {
      try {
        std::thread{[adapter] { delete adapter; }}.detach();
        return;
      } catch (...) {
        std::terminate();
      }
    }
    delete adapter;
  };
  auto adapter = std::shared_ptr<VulkanAdapter>{
      new VulkanAdapter{HasVulkanExtension(created.extensions,
                                           kVulkanPortabilitySubsetExtension)},
      destroy};
  adapter->instance = instance;
  adapter->physical_device = physical_device;
  vkGetPhysicalDeviceMemoryProperties(physical_device,
                                      &adapter->memory_properties);
  adapter->device = created.device;
  adapter->compute_queue = created.queue;
  adapter->compute_queue_family = created.queue_family;
  adapter->persistent_stream_submit_capacity =
      persistent_stream_submit_capacity;
  adapter->timestamp_valid_bits = created.timestamp_valid_bits;
  static_cast<void>(CreateVulkanTimeline(created.device, created.timeline,
                                         adapter->timeline));
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
  // Publish the worker lifetime token before either completion thread can
  // read it. AccelDeviceFromVulkanAdapter later projects this same owner.
  adapter->owner_token = adapter;
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
