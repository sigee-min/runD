#pragma once

#include <accel/check.hpp>

#include "buffers.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PrepareMetalGatherPipeline(MetalAdapter &adapter,
                           const rund::kernel::GatherPlan &plan,
                           MetalGatherEncodeResources &resources) {
  if (CompileMetalGatherPipelines(adapter, plan.element,
                                  resources.control_pipeline,
                                  resources.gather_pipeline)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
