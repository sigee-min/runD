#pragma once

#include <accel/check.hpp>

#include "lookup.hpp"

#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PrepareMetalScatterStatusBuffer(MetalAdapter &adapter,
                                const rund::kernel::ScatterPlan &plan,
                                MetalScatterEncodeResources &resources) {
  resources.status = AcquireMetalBuffer(adapter, plan.status_bytes,
                                        MetalBufferUsage::Output);
  void *const contents = MetalBufferContents(resources.status);
  if (contents == nullptr) {
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  std::memset(contents, 0xff, static_cast<std::size_t>(plan.status_bytes));
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
