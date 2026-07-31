#include <accel/check.hpp>

#include "../local.hpp"

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
#if defined(__APPLE__)
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
#endif

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "runD node Vulkan discovery";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "runD";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
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
                            .instance = instance};
}
#endif

} // namespace rund::node::accel::detail
