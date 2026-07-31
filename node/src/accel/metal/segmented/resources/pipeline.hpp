#pragma once

#include <accel/check.hpp>

#include "buffers.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalSegmentedScanPipeline(
    MetalAdapter &adapter, const rund::kernel::SegmentedScanPlan &plan,
    const rund::kernel::ComputeDomain domain,
    MetalSegmentedScanEncodeResources &resources) {
  if (CompileMetalSegmentedScanPipelines(adapter, plan.element, domain,
                                         resources.block, resources.prefix,
                                         resources.offset)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
