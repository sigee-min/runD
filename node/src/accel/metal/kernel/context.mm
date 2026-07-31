#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck ValidateMetalKernelContext(const rund::AccelDevice &pick,
                                            MetalKernelContext &out) {
  if (pick.api != rund::AccelApi::Metal || pick.backend.context == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  out.adapter = adapter;
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
