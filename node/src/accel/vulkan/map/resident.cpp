#include "../../backend/number.hpp"
#include "../descriptor.hpp"
#include "local.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool VulkanMapResidentWindowSpan(
    const VulkanAdapter &adapter, const rund::kernel::ResidentBufferRef &ref,
    const rund::kernel::ComputeDispatchWindow &window, VkDeviceSize &offset,
    VkDeviceSize &range, const char *&reason) noexcept {
  if (ref.stride_bytes < ref.element_bytes) {
    reason = "compute_resident_stride_invalid";
    return false;
  }
  StorageRange planned{};
  if (!PlanStorage(adapter, ref, window.begin_sequence, window.tile_count,
                   planned)) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  offset = planned.base;
  range = planned.bytes;
  reason = "ok";
  return true;
}
#endif

} // namespace rund::node::accel::detail
