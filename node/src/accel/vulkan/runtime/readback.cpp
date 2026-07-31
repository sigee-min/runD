#include "local.hpp"
#include <rund/counter.hpp>

#include <cstddef>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool FinishVulkanWindowReadback(
    VulkanAdapter& adapter,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanWindowBuffers& buffers) {
  if (buffers.resident) { return true; }
  if (buffers.output.buffer.mapped == nullptr ||
      buffers.output.used_bytes < static_cast<VkDeviceSize>(
                                    buffers.output_size)) {
    SetVulkanLastError(adapter, "accel_vulkan_readback_failed");
    return false;
  }
  rund::kernel::u64 copied = 0u;
  if (!ScatterOutputBytes(
          bindings, window,
          static_cast<const std::byte*>(buffers.output.buffer.mapped),
          buffers.output_size, buffers.staged, copied)) {
    SetVulkanLastError(adapter, "accel_vulkan_readback_failed");
    return false;
  }
  ::rund::detail::counter::Accumulate(adapter.device_to_host_bytes, copied);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
