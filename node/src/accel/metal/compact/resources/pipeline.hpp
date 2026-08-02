#pragma once

#include <accel/check.hpp>

#include "../../scan/pipeline.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
CompileMetalCompactPipelineSet(MetalAdapter &adapter,
                               MetalCompactEncodeResources &resources) {
  // Preserve the established cold-admission priority: Compact's own
  // executables are admitted before the internal Scan dependency.  Both are
  // retained here so the warm encoder only borrows immutable handles.
  if (!CompileMetalCompactPipelines(adapter, resources.pipelines,
                                    resources.plan.status_bytes != 0u,
                                    resources.block_offset_path)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  const bool scan_ready =
      resources.block_offset_path
          ? CompileMetalScanPipelines(adapter, resources.scan_plan.element,
                                      resources.scan_block,
                                      resources.scan_prefix,
                                      resources.scan_offset)
          : CompileMetalScanFlagPipelines(
                adapter, resources.scan_block, resources.scan_prefix,
                resources.scan_offset);
  if (scan_ready) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
  return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
