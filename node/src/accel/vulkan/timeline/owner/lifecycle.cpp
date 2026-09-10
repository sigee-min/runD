#include "internal.hpp"

#include "../../local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

VulkanTimelineSupport QueryVulkanTimelineSupport(
    const VkPhysicalDevice physical_device,
    const std::uint32_t instance_api_version,
    const std::vector<VkExtensionProperties> &extensions) noexcept {
  if (physical_device == VK_NULL_HANDLE) {
    return {};
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  const std::uint32_t effective_api =
      std::min(instance_api_version, properties.apiVersion);
  const bool core = VK_VERSION_MAJOR(effective_api) > 1u ||
                    (VK_VERSION_MAJOR(effective_api) == 1u &&
                     VK_VERSION_MINOR(effective_api) >= 2u);
  const bool extension =
      HasVulkanExtension(extensions, kVulkanTimelineExtension);
  if (!core && !extension) {
    return {};
  }
  VkPhysicalDeviceFeatures2 features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  VkPhysicalDeviceVulkan12Features core_features{};
  VkPhysicalDeviceVulkan12Properties core_properties{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR extension_features{};
  VkPhysicalDeviceTimelineSemaphorePropertiesKHR extension_properties{};
  std::uint64_t max_difference = 0u;
  if (core) {
    core_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features.pNext = &core_features;
    core_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    properties2.pNext = &core_properties;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    if (core_features.timelineSemaphore != VK_TRUE) {
      return {};
    }
    max_difference = core_properties.maxTimelineSemaphoreValueDifference;
  } else {
    extension_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
    features.pNext = &extension_features;
    extension_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR;
    properties2.pNext = &extension_properties;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    if (extension_features.timelineSemaphore != VK_TRUE) {
      return {};
    }
    max_difference = extension_properties.maxTimelineSemaphoreValueDifference;
  }
  if (max_difference == 0u) {
    return {};
  }
  return VulkanTimelineSupport{.route = core ? VulkanTimelineRoute::Core
                                             : VulkanTimelineRoute::Extension,
                               .max_difference = max_difference};
}

rund::AccelCheck CreateVulkanTimeline(const VkDevice device,
                                      const VulkanTimelineSupport support,
                                      VulkanTimelineOwner &owner) noexcept {
  DestroyVulkanTimeline(owner);
  const auto reject = [&](const rund::AccelCheck check) noexcept {
    DestroyVulkanTimeline(owner);
    timeline_owner::record_failure(owner, check);
    return check;
  };
  if (!support || support.max_difference == 0u) {
    return reject(timeline_owner::unsupported());
  }
  if (device == VK_NULL_HANDLE) {
    return reject(timeline_owner::invalid());
  }
  owner.device = device;
  owner.support = support;
  owner.signal = reinterpret_cast<PFN_vkSignalSemaphoreKHR>(
      timeline_owner::load(device, support.route, "vkSignalSemaphore",
                           "vkSignalSemaphoreKHR"));
  owner.wait = reinterpret_cast<PFN_vkWaitSemaphoresKHR>(
      timeline_owner::load(device, support.route, "vkWaitSemaphores",
                           "vkWaitSemaphoresKHR"));
  owner.counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValueKHR>(
      timeline_owner::load(device, support.route, "vkGetSemaphoreCounterValue",
                           "vkGetSemaphoreCounterValueKHR"));
  if (owner.signal == nullptr || owner.wait == nullptr ||
      owner.counter == nullptr) {
    return reject(timeline_owner::failed());
  }
  VkSemaphoreTypeCreateInfoKHR type{};
  type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
  type.initialValue = 0u;
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = &type;
  for (VkSemaphore &ready : owner.ready) {
    if (vkCreateSemaphore(device, &info, nullptr, &ready) != VK_SUCCESS ||
        ready == VK_NULL_HANDLE) {
      return reject(timeline_owner::failed());
    }
  }
  if (vkCreateSemaphore(device, &info, nullptr, &owner.done) != VK_SUCCESS ||
      owner.done == VK_NULL_HANDLE) {
    return reject(timeline_owner::failed());
  }
  owner.init_state = VulkanTimelineInitState::Initialized;
  owner.init_check = rund::AccelCheck{true, "ok"};
  return owner.init_check;
}

void DestroyVulkanTimeline(VulkanTimelineOwner &owner) noexcept {
  if (owner.device != VK_NULL_HANDLE) {
    if (owner.done != VK_NULL_HANDLE) {
      vkDestroySemaphore(owner.device, owner.done, nullptr);
    }
    for (VkSemaphore &ready : owner.ready) {
      if (ready != VK_NULL_HANDLE) {
        vkDestroySemaphore(owner.device, ready, nullptr);
      }
    }
  }
  owner.device = VK_NULL_HANDLE;
  owner.ready.fill(VK_NULL_HANDLE);
  owner.done = VK_NULL_HANDLE;
  owner.signal = nullptr;
  owner.wait = nullptr;
  owner.counter = nullptr;
  owner.support = {};
  owner.ready_signaled.fill(0u);
  owner.ready_wait_scheduled.fill(0u);
  owner.next_ready_value.fill(1u);
  owner.generation_ready_value.fill(0u);
  owner.generation_first_ready_value.fill(0u);
  owner.ordered_ready_signaled = 0u;
  owner.done_signal_scheduled = 0u;
  owner.next_generation = 1u;
  owner.open_generation = 0u;
  owner.next_value = 1u;
  owner.generation_first_value = 0u;
  owner.generation_last_value = 0u;
  owner.terminal_loss = {};
  owner.window_ready_mode = false;
  owner.init_state = VulkanTimelineInitState::Uninitialized;
  owner.init_check = timeline_owner::unsupported();
}

#endif

} // namespace rund::node::accel::detail
