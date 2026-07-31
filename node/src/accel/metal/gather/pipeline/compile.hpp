#pragma once

#include "cache.hpp"
#include "../../pipeline/named.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalGatherPipelineLibrary(
    MetalAdapter& adapter,
    const rund::kernel::GatherElement element,
    std::shared_ptr<void>& control,
    std::shared_ptr<void>& gather) {
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalGatherSource());
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  return library != nil &&
         MakeNamedMetalPipeline(device, library, "rund_compute_gather_control",
                                control) &&
         MakeNamedMetalPipeline(device, library, GatherFunctionName(element),
                                gather);
}
#endif

}  // namespace rund::node::accel::detail
