#pragma once

#include <accel/check.hpp>

#include "buffers.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PrepareMetalReducePipeline(MetalAdapter &adapter,
                           const rund::kernel::ReducePlan &plan,
                           const rund::kernel::ComputeDomain domain,
                           MetalReduceEncodeResources &resources) {
  if (CompileMetalReducePipeline(adapter, plan, domain, resources.pipeline)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
