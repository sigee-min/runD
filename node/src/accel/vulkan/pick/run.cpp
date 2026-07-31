#include <accel/device.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelDevice PickWithVulkanSdk() {
#if !defined(RUND_NODE_HAVE_GLSLANG_VALIDATOR)
  return RejectVulkan("accel_vulkan_shader_tool_unavailable");
#else
  const VulkanInstancePick instance = CreateVulkanDiscoveryInstance();
  if (!instance.check.ok) {
    return RejectVulkan(instance.check.reason);
  }

  const VulkanAdapterPick adapter =
      PickVulkanAdapterFromInstance(instance.instance);
  if (!adapter.check.ok) {
    vkDestroyInstance(instance.instance, nullptr);
    return RejectVulkan(adapter.check.reason);
  }
  return AccelDeviceFromVulkanAdapter(adapter.adapter);
#endif
}
#endif

rund::AccelDevice PickVulkan() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return PickWithVulkanSdk();
#else
  return RejectVulkan("accel_vulkan_loader_unavailable");
#endif
}

} // namespace rund::node::accel::detail
