#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
AcquireMetalScanDirectBuffers(MetalAdapter &adapter,
                              const rund::kernel::ScanPlan &plan,
                              MetalScanDirectBuffers &buffers) {
  buffers.totals = AcquireMetalBuffer(
      adapter, plan.block_count * plan.element_bytes, MetalBufferUsage::Scratch);
  buffers.status = AcquireMetalBuffer(adapter, sizeof(rund::kernel::u32),
                                      MetalBufferUsage::Output);
  if (buffers.totals.buffer != nullptr && buffers.status.buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  ReleaseMetalScanDirectBuffers(adapter, buffers);
  SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
  return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
