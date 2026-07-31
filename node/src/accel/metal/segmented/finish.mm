#include <accel/check.hpp>

#include "../../segmented/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck
FinishMetalSegmentedScan(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const scan =
      static_cast<MetalSegmentedScanEncodeResources *>(resources.get());
  if (scan == nullptr || scan->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  const auto *const status =
      static_cast<const rund::kernel::u32 *>(MetalBufferContents(scan->status));
  if (status == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  const rund::AccelCheck check = SegmentedScanStatus(*status);
  if (!check.ok) {
    SetMetalLastError(adapter, check.reason);
    return check;
  }
  RecordMetalDispatches(adapter, EncodedSegmentedScanDispatchCount(scan->plan));
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
