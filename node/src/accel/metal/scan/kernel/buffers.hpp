#pragma once

#include <accel/check.hpp>

#include "../../../scan/prefix.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
AcquireMetalScanScratch(MetalAdapter &adapter,
                        const rund::kernel::ScanPlan &plan,
                        MetalScanEncodeResources &resources) {
  const auto totals_bytes = ScanPrefixTotalsBytes(plan);
  if (!totals_bytes.has_value()) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  resources.totals =
      AcquireMetalBuffer(adapter, *totals_bytes, MetalBufferUsage::Scratch);
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
