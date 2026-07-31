#include <accel/check.hpp>

#include "../../scan/count.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalPartition(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const partition =
      static_cast<MetalPartitionEncodeResources *>(resources.get());
  if (partition == nullptr || partition->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  if (!MetalScanStatusOk(partition->false_status)) {
    SetMetalLastError(adapter, "compute_scan_sum_overflow");
    return rund::AccelCheck{false, "compute_scan_sum_overflow"};
  }
  const rund::kernel::u64 dispatch_count =
      2u + EncodedScanDispatchCount(partition->scan_plan);
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
