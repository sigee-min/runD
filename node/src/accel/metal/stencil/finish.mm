#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalStencil(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const stencil =
      static_cast<MetalStencilEncodeResources *>(resources.get());
  if (stencil == nullptr || stencil->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  RecordMetalDispatches(adapter, stencil->plan.pass_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
