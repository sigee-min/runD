#pragma once

#include "../../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool
ResidentWindowOffset(const rund::kernel::ResidentBufferRef &ref,
                     const rund::kernel::ComputeDispatchWindow &window,
                     NSUInteger &offset, rund::kernel::u64 &range) noexcept {
  if (ref.stride_bytes < ref.element_bytes || window.tile_count == 0u ||
      !rund::kernel::checked::mul(window.begin_sequence, ref.stride_bytes) ||
      !rund::kernel::checked::mul(window.tile_count - 1u, ref.stride_bytes)) {
    return false;
  }
  const rund::kernel::u64 relative = window.begin_sequence * ref.stride_bytes;
  if (!rund::kernel::checked::add(ref.offset_bytes, relative)) {
    return false;
  }
  const rund::kernel::u64 byte_offset = ref.offset_bytes + relative;
  const rund::kernel::u64 tail = (window.tile_count - 1u) * ref.stride_bytes;
  if (!rund::kernel::checked::add(tail, ref.element_bytes)) {
    return false;
  }
  range = tail + ref.element_bytes;
  if (!rund::kernel::checked::add(byte_offset, range) ||
      byte_offset + range > ref.bytes) {
    return false;
  }
  return ToNSUInteger(byte_offset, offset);
}
#endif

} // namespace rund::node::accel::detail
