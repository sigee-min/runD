#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
CompileMetalSortPipelineSet(MetalAdapter &adapter,
                            MetalSortEncodeResources &resources) {
  if (CompileMetalSortPipelines(adapter, resources.plan.key,
                                resources.block_size, resources.pipelines)) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
