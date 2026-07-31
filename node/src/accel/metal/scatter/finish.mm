#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalScatter(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const scatter =
      static_cast<MetalScatterEncodeResources *>(resources.get());
  if (scatter == nullptr || scatter->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_scatter_invalid");
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  const auto *const status = static_cast<const rund::kernel::u32 *>(
      MetalBufferContents(scatter->status));
  if (status == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (*status != kScatterStatusOk) {
    const rund::kernel::u32 reason = *status & 1u;
    const char *const failure = reason == kScatterReasonOutOfRange
                                    ? "compute_scatter_index_out_of_range"
                                    : (reason == kScatterReasonDuplicate
                                           ? "compute_scatter_duplicate_index"
                                           : "compute_scatter_invalid");
    SetMetalLastError(adapter, failure);
    return rund::AccelCheck{false, failure};
  }
  RecordMetalDispatches(adapter, scatter->plan.pass_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
