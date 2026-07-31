#include "../pipeline/named.hpp"
#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

bool CompileMetalHistogramPipelines(MetalAdapter &adapter,
                                    MetalHistogramPipelines &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.clear = LookupMetalNamedPipeline(adapter, "histogram.clear.u32");
  out.count = LookupMetalNamedPipeline(adapter, "histogram.count.u32");
  if (out.clear != nullptr && out.count != nullptr) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalHistogramSource());
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library, "rund_compute_histogram_clear",
                              out.clear) ||
      !MakeNamedMetalPipeline(device, library, "rund_compute_histogram_count",
                              out.count)) {
    return false;
  }
  StoreMetalNamedPipeline(adapter, "histogram.clear.u32", out.clear,
                          MonotonicNanoseconds() - begin);
  StoreMetalNamedPipeline(adapter, "histogram.count.u32", out.count, 0u);
  return true;
#else
  (void)adapter;
  (void)out;
  return false;
#endif
}

} // namespace rund::node::accel::detail
