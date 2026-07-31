#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelDevice
AccelDeviceFromVulkanAdapter(const std::shared_ptr<VulkanAdapter> &adapter) {
  std::shared_ptr<void> owner = adapter;
  adapter->owner_token = owner;
  return rund::AccelDevice{
      .check = rund::AccelCheck{true, "ok"},
      .api = rund::AccelApi::Vulkan,
      .caps = adapter->caps,
      .backend =
          rund::kernel::ComputeBackendDispatch{
              .context = adapter.get(),
              .execute = ExecuteVulkan,
              .last_error = VulkanLastError,
          },
      .backend_info =
          rund::AccelBackendInfo{
              .device_name = adapter->device_name,
              .driver_name = adapter->driver_name,
              .driver_info = adapter->driver_info,
              .storage_alignment = adapter->storage_align,
              .storage_bytes = adapter->storage_limit,
          },
      .owner = owner,
  };
}
#endif

} // namespace rund::node::accel::detail
