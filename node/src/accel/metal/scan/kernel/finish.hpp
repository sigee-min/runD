#pragma once

#include <accel/check.hpp>

#include "../../../scan/count.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalScan(MetalAdapter &adapter,
                                 const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const scan = static_cast<MetalScanEncodeResources *>(resources.get());
  if (scan == nullptr || scan->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  const rund::kernel::u32 flags = MetalScanStatusFlags(scan->status);
  if (flags != 0u) {
    const char *const reason = (flags & 2u) != 0u
                                   ? "compute_bounded_count_invalid"
                                   : "compute_scan_sum_overflow";
    SetMetalLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  RecordMetalDispatches(adapter, EncodedScanDispatchCount(scan->plan));
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
