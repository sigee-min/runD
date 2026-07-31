#include "../../scan/vulkan.hpp"
#include "../adapter/api.hpp"
#include "../status.hpp"

namespace rund::node::accel::detail {

rund::kernel::u32 VulkanScanStatusFlags(const VulkanStatus &status) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const values = VulkanStatusValue(status);
  if (values == nullptr) {
    return ~rund::kernel::u32{0u};
  }
  return values[0];
#else
  (void)status;
  return ~rund::kernel::u32{0u};
#endif
}

bool VulkanScanStatusOk(const VulkanStatus &status) {
  return VulkanScanStatusFlags(status) == 0u;
}

} // namespace rund::node::accel::detail
