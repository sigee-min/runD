#include <accel/context/buffer/descriptor.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

::rund::AccelCheck CheckDesc(const rund::AccelBufferDesc &desc) noexcept {
  if (!KnownUsage(desc.usage) || desc.scalar_width_bytes == 0u ||
      desc.count == 0u) {
    return RejectAccelCheck("accel_context_buffer_invalid");
  }
  if (!rund::kernel::checked::mul(desc.scalar_width_bytes, desc.count)) {
    return RejectAccelCheck("accel_context_buffer_overflow");
  }
  return OkAccelCheck();
}

} // namespace rund::node::accel::detail
