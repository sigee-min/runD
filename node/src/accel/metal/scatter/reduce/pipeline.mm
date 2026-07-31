#include "model.hpp"

#include "../../../clock.hpp"
#include "../../pipeline/cache.hpp"
#include "../../pipeline/named.hpp"

#include <cstdint>
#include <memory>
#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool AcquireMetalScatterReducePipelines(
    MetalAdapter &adapter, const rund::kernel::ScatterReducePlan &plan,
    std::shared_ptr<void> &control, std::shared_ptr<void> &init,
    std::shared_ptr<void> &fold) {
  const std::string key = MetalScatterReduceKey(plan);
  control = LookupMetalNamedPipeline(adapter, key + ".control");
  init = LookupMetalNamedPipeline(adapter, key + ".init");
  fold = LookupMetalNamedPipeline(adapter, key + ".fold");
  if (control != nullptr && init != nullptr && fold != nullptr) {
    return true;
  }
  const std::uint64_t begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalScatterReduceSource(plan));
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library, "rund_scatter_reduce_control",
                              control) ||
      !MakeNamedMetalPipeline(device, library, "rund_scatter_reduce_initialize",
                              init) ||
      !MakeNamedMetalPipeline(device, library,
                              "rund_scatter_reduce_fold_sources", fold)) {
    return false;
  }
  const std::uint64_t elapsed = MonotonicNanoseconds() - begin;
  StoreMetalNamedPipeline(adapter, key + ".control", control, elapsed);
  StoreMetalNamedPipeline(adapter, key + ".init", init, 0u);
  StoreMetalNamedPipeline(adapter, key + ".fold", fold, 0u);
  return true;
}

#endif

} // namespace rund::node::accel::detail
