#include "../local.hpp"

#include <algorithm>
#include <cstddef>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool MakeHostBuffer(VulkanAdapter& adapter,
                    const void* const bytes,
                    const rund::kernel::u64 byte_count,
                    ScopedBuffer& out) {
  VulkanBuffer buffer{};
  const VkDeviceSize used_bytes =
      static_cast<VkDeviceSize>(std::max<rund::kernel::u64>(4u, byte_count));
  if (!CreateVulkanBuffer(adapter,
                              used_bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              buffer)) {
    return false;
  }
  out = ScopedBuffer{adapter, buffer, used_bytes};
  if (byte_count == 0u) {
    return ClearVulkanBuffer(out.buffer, used_bytes);
  }
  return UploadVulkanBuffer(out.buffer, bytes,
                            static_cast<VkDeviceSize>(byte_count));
}
#endif

}  // namespace rund::node::accel::detail
