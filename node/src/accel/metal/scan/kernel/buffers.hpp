#pragma once

#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
AcquireMetalScanScratch(MetalAdapter &adapter,
                        MetalScanEncodeResources &resources) {
  if (!resources.prefix_execution.has_value() ||
      resources.prefix_execution->temporary_count() != 1u) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  resources.totals = AcquireMetalBuffer(
      adapter, resources.prefix_execution->temporary(0u).bytes,
      MetalBufferUsage::Scratch);
  resources.status = AcquireMetalBuffer(adapter, sizeof(rund::kernel::u32),
                                        MetalBufferUsage::Output);
  if (resources.totals.buffer != nullptr &&
      resources.status.buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
  return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
