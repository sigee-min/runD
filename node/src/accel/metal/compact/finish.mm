#include <accel/check.hpp>

#include "../../scan/count.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalCompact(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const compact =
      static_cast<MetalCompactEncodeResources *>(resources.get());
  if (compact == nullptr || compact->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_compact_invalid");
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  if (!MetalScanStatusOk(compact->scan_status)) {
    SetMetalLastError(adapter, "compute_scan_sum_overflow");
    return rund::AccelCheck{false, "compute_scan_sum_overflow"};
  }
  if (compact->plan.status_bytes != 0u) {
    const auto *const selected_count = static_cast<const rund::kernel::u32 *>(
        MetalBufferContents(compact->status));
    if (selected_count == nullptr) {
      SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (static_cast<rund::kernel::u64>(*selected_count) >
        compact->plan.output_capacity) {
      SetMetalLastError(adapter, "compute_compact_capacity_insufficient");
      return rund::AccelCheck{false, "compute_compact_capacity_insufficient"};
    }
  }
  const rund::kernel::u64 dispatch_count =
      compact->block_offset_path
          ? 2u + EncodedScanDeferredOffsetDispatchCount(compact->scan_plan)
          : EncodedScanDeferredOffsetDispatchCount(compact->scan_plan) + 1u +
                (compact->plan.status_bytes != 0u ? 1u : 0u);
  RecordMetalDispatches(adapter, dispatch_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
