#include "../pipeline/named.hpp"
#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

bool CompileMetalHistogramPipelines(MetalAdapter &adapter,
                                    const std::uint64_t bins,
                                    MetalHistogramPipelines &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const char *const count_key = MetalHistogramLocal(bins)
                                    ? "histogram.count.local.u32"
                                    : "histogram.count.cohort.u32";
  out.clear = LookupMetalNamedPipeline(adapter, "histogram.clear.u32");
  out.count = LookupMetalNamedPipeline(adapter, count_key);
  if (out.clear != nullptr && out.count != nullptr) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalHistogramSource(bins));
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  const bool create_clear = out.clear == nullptr;
  if (library == nil ||
      (create_clear &&
       !MakeNamedMetalPipeline(device, library, "rund_compute_histogram_clear",
                               out.clear)) ||
      !MakeNamedMetalPipeline(device, library, "rund_compute_histogram_count",
                              out.count)) {
    return false;
  }
  if (create_clear) {
    StoreMetalNamedPipeline(adapter, "histogram.clear.u32", out.clear, 0u);
  }
  StoreMetalNamedPipeline(adapter, count_key, out.count,
                          MonotonicNanoseconds() - begin);
  return true;
#else
  (void)adapter;
  (void)bins;
  (void)out;
  return false;
#endif
}

} // namespace rund::node::accel::detail
