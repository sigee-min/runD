#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool ResidentWindowSpan(const rund::kernel::ResidentBufferRef &ref,
                        const rund::kernel::ComputeDispatchWindow &window,
                        VkDeviceSize &offset, VkDeviceSize &range,
                        const char *&reason) noexcept {
  if (ref.stride_bytes != ref.element_bytes) {
    reason = "compute_resident_stride_invalid";
    return false;
  }
  if (!rund::kernel::checked::mul(window.begin_sequence, ref.element_bytes) ||
      !rund::kernel::checked::mul(window.tile_count, ref.element_bytes)) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  const rund::kernel::u64 byte_offset =
      window.begin_sequence * ref.element_bytes;
  const rund::kernel::u64 byte_range = window.tile_count * ref.element_bytes;
  if (!rund::kernel::checked::add(byte_offset, byte_range) ||
      byte_offset + byte_range > ref.bytes) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  offset = static_cast<VkDeviceSize>(byte_offset);
  range = static_cast<VkDeviceSize>(byte_range);
  reason = "ok";
  return true;
}
#endif

} // namespace rund::node::accel::detail
