#pragma once

#include <accel/check.hpp>

#include "lookup.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PrepareMetalStencilPipeline(MetalAdapter &adapter,
                            const rund::kernel::StencilPlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            MetalStencilEncodeResources &resources) {
  if (CompileMetalStencilPipeline(adapter, plan.op, plan.element, domain,
                                  resources.pipeline)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
