#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "adapter/state.hpp"
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice RejectVulkan(const char *reason);

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint64_t kVulkanOneMiB = 1024u * 1024u;
inline constexpr const char *kVulkanPortabilityEnumerationExtension =
    "VK_KHR_portability_enumeration";
inline constexpr const char *kVulkanPortabilitySubsetExtension =
    "VK_KHR_portability_subset";
inline constexpr const char *kVulkanDriverPropertiesExtension =
    "VK_KHR_driver_properties";

struct VulkanInstancePick {
  rund::AccelCheck check{};
  VkInstance instance = VK_NULL_HANDLE;
  std::uint32_t api_version = VK_API_VERSION_1_1;
  std::uint64_t persistent_stream_submit_capacity{};
};

struct VulkanAdapterPick {
  rund::AccelCheck check{};
  std::shared_ptr<VulkanAdapter> adapter{};
};

[[nodiscard]] rund::AccelDevice PickWithVulkanSdk();
[[nodiscard]] VulkanInstancePick CreateVulkanDiscoveryInstance();
[[nodiscard]] VulkanAdapterPick
PickVulkanAdapterFromInstance(VkInstance instance,
                              std::uint32_t instance_api_version,
                              std::uint64_t persistent_stream_submit_capacity);
[[nodiscard]] rund::AccelDevice
AccelDeviceFromVulkanAdapter(const std::shared_ptr<VulkanAdapter> &adapter);

[[nodiscard]] bool
HasVulkanExtension(const std::vector<VkExtensionProperties> &props,
                   std::string_view name);
[[nodiscard]] bool
EnumerateVulkanInstanceExtensions(std::vector<VkExtensionProperties> &props);
[[nodiscard]] bool
EnumerateVulkanDeviceExtensions(VkPhysicalDevice physical_device,
                                std::vector<VkExtensionProperties> &props);

[[nodiscard]] bool
FindVulkanComputeQueueFamily(VkPhysicalDevice physical_device,
                             std::uint32_t &queue_family,
                             std::uint32_t &timestamp_valid_bits);
[[nodiscard]] std::uint64_t VulkanDeviceMemoryBytes(
    VkPhysicalDevice physical_device,
    const VkPhysicalDeviceMemoryProperties &properties,
    const std::vector<VkExtensionProperties> &device_extensions);
void PopulateVulkanDriverProperties(
    VkInstance instance, const VkPhysicalDeviceProperties &properties,
    bool driver_properties_available, VulkanAdapter &adapter);
[[nodiscard]] bool VulkanDeviceSupportsDriverProperties(
    VkPhysicalDevice physical_device,
    const std::vector<VkExtensionProperties> &device_extensions);
[[nodiscard]] bool
VulkanDeviceSupportsRequiredFeatures(VkPhysicalDevice physical_device,
                                     VkPhysicalDeviceFeatures &features);

void TryCreateVulkanTimestampQueryPool(VulkanAdapter &adapter);
#endif

} // namespace rund::node::accel::detail
