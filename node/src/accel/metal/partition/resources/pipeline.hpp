#pragma once

#include <accel/check.hpp>

#include "../../scan/pipeline.hpp"
#include "buffers.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LoadMetalPartitionPipelines(MetalAdapter &adapter,
                            MetalPartitionEncodeResources &raw) {
  if (!CompileMetalPartitionPipelines(adapter, raw.plan.flag_bytes,
                                      raw.plan.value_bytes, raw.pipelines)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  if (!CompileMetalScanPipelines(adapter, raw.scan_plan.element, raw.scan_block,
                                 raw.scan_prefix, raw.scan_offset)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
