#include <accel/check.hpp>

#include "local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalRange(MetalAdapter &adapter,
                                  const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const range = static_cast<MetalRangeResources *>(resources.get());
  if (range == nullptr || range->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  RecordMetalDispatches(adapter, range->range.stage_count());
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
