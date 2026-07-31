#pragma once

#include "../../pipeline/named.hpp"
#include "cache.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalReducePipelineLibrary(
    MetalAdapter &adapter, const rund::kernel::ReducePlan &plan,
    const rund::kernel::ComputeDomain domain, std::shared_ptr<void> &out) {
  std::shared_ptr<void> library_owner = AcquireMetalLibrary(
      adapter, MetalReduceSource(plan.op, plan.block_size, domain));
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  return library != nil &&
         MakeNamedMetalPipeline(device, library,
                                ReduceFunctionName(plan.op, plan.element), out);
}
#endif

} // namespace rund::node::accel::detail
