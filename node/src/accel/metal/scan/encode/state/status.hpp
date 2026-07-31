#pragma once

#include <accel/check.hpp>

#include "check.hpp"
#include "../../../../kernel/preparation.hpp"

#include <cstddef>
#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
ResetMetalScanStatusBuffer(MetalAdapter &adapter, void *const status_buffer) {
  if (IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    return rund::AccelCheck{true, "ok"};
  }
  id<MTLBuffer> status_mtl = (__bridge id<MTLBuffer>)status_buffer;
  if (status_mtl == nil || [status_mtl contents] == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  std::memset([status_mtl contents], 0, sizeof(rund::kernel::u32));
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
