#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalHistogram(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const histogram =
      static_cast<MetalHistogramEncodeResources *>(resources.get());
  if (histogram == nullptr || histogram->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  const auto *const status = static_cast<const rund::kernel::u32 *>(
      MetalBufferContents(histogram->status));
  if (status == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (*status != kHistogramStatusOk) {
    SetMetalLastError(adapter, "compute_histogram_bin_invalid");
    return rund::AccelCheck{false, "compute_histogram_bin_invalid"};
  }
  RecordMetalDispatches(adapter, histogram->plan.pass_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
