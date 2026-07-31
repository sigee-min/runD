#pragma once

#include <accel/check.hpp>

#include "../local.hpp"
#include "../../../kernel/preparation.hpp"

#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
ResetMetalScatterStatus(MetalAdapter &adapter,
                        MetalScatterEncodeResources &scatter) {
  if (IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    return rund::AccelCheck{true, "ok"};
  }
  void *const contents = MetalBufferContents(scatter.status);
  if (contents == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  std::memset(contents, 0xff,
              static_cast<std::size_t>(scatter.plan.status_bytes));
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
