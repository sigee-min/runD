#pragma once

#include "../../pipeline/named.hpp"
#include "cache.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalStencilPipelineLibrary(
    MetalAdapter &adapter, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain, std::shared_ptr<void> &out) {
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalStencilSource(op));
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  const std::string function_name = StencilFunctionName(op, element, domain);
  return library != nil &&
         MakeNamedMetalPipeline(device, library, function_name.c_str(), out);
}
#endif

} // namespace rund::node::accel::detail
