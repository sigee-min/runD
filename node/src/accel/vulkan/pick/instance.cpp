#include <accel/check.hpp>

#include "../local.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanInstancePick CreateVulkanDiscoveryInstance() {
  std::vector<VkExtensionProperties> instance_props{};
  if (!EnumerateVulkanInstanceExtensions(instance_props)) {
    return VulkanInstancePick{
        .check = rund::AccelCheck{false, "accel_vulkan_loader_unavailable"}};
  }

  std::vector<const char *> instance_extensions{};
  VkInstanceCreateFlags instance_flags = 0u;
  std::uint64_t persistent_stream_submit_capacity =
      std::numeric_limits<std::uint64_t>::max();
#if defined(__APPLE__) && defined(VK_EXT_layer_settings)
  constexpr std::uint32_t persistent_metal_command_capacity = 1024u;
  const VkLayerSettingEXT persistent_stream_setting{
      .pLayerName = "MoltenVK",
      .pSettingName = "MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE",
      .type = VK_LAYER_SETTING_TYPE_UINT32_EXT,
      .valueCount = 1u,
      .pValues = &persistent_metal_command_capacity,
  };
  VkLayerSettingsCreateInfoEXT layer_settings{
      .sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
      .settingCount = 1u,
      .pSettings = &persistent_stream_setting,
  };
#endif
#if defined(__APPLE__)
  persistent_stream_submit_capacity = 0u;
  if (!HasVulkanExtension(instance_props,
                          kVulkanPortabilityEnumerationExtension)) {
    return VulkanInstancePick{
        .check =
            rund::AccelCheck{false, "accel_vulkan_portability_unavailable"}};
  }
  instance_extensions.push_back(kVulkanPortabilityEnumerationExtension);
#if defined(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
  instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
  instance_flags |= 0x00000001u;
#endif
#if defined(VK_EXT_layer_settings)
  if (HasVulkanExtension(instance_props,
                         VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)) {
    instance_extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    persistent_stream_submit_capacity = persistent_metal_command_capacity;
  }
#endif
#endif

  std::uint32_t loader_api = VK_API_VERSION_1_1;
  const auto enumerate_version =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerate_version != nullptr) {
    std::uint32_t reported = VK_API_VERSION_1_1;
    if (enumerate_version(&reported) == VK_SUCCESS) {
      loader_api = reported;
    }
  }
  const std::uint32_t requested_api = loader_api >= VK_API_VERSION_1_2
                                          ? VK_API_VERSION_1_2
                                          : VK_API_VERSION_1_1;

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "runD node Vulkan discovery";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "runD";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = requested_api;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pNext =
#if defined(__APPLE__) && defined(VK_EXT_layer_settings)
      persistent_stream_submit_capacity != 0u ? &layer_settings : nullptr;
#else
      nullptr;
#endif
  instance_info.flags = instance_flags;
  instance_info.pApplicationInfo = &app_info;
  instance_info.enabledExtensionCount =
      static_cast<std::uint32_t>(instance_extensions.size());
  instance_info.ppEnabledExtensionNames = instance_extensions.data();

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS ||
      instance == VK_NULL_HANDLE) {
    return VulkanInstancePick{
        .check = rund::AccelCheck{false, "accel_vulkan_instance_unavailable"}};
  }
  return VulkanInstancePick{.check = rund::AccelCheck{true, "ok"},
                            .instance = instance,
                            .api_version = requested_api,
                            .persistent_stream_submit_capacity =
                                persistent_stream_submit_capacity};
}
#endif

} // namespace rund::node::accel::detail
