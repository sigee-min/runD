#pragma once

#include "cache.hpp"
#include "../../pipeline/named.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalScatterPipelineLibrary(
    MetalAdapter& adapter,
    const rund::kernel::ScatterElement element,
    std::shared_ptr<void>& out) {
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalScatterSource());
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  return library != nil &&
         MakeNamedMetalPipeline(device, library, ScatterFunctionName(element),
                                out);
}
#endif

}  // namespace rund::node::accel::detail
