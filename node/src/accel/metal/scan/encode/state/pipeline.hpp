#pragma once

#include <accel/check.hpp>

#include "status.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LoadMetalScanPipelines(MetalAdapter &adapter,
                       const rund::kernel::ScanPlan &plan,
                       MetalScanEncodeState &state) {
  if (!CompileMetalScanPipelines(adapter, plan.element, state.block_handle,
                                 state.prefix_handle, state.offset_handle)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  state.block = (__bridge id<MTLComputePipelineState>)state.block_handle.get();
  state.prefix =
      (__bridge id<MTLComputePipelineState>)state.prefix_handle.get();
  state.offset =
      (__bridge id<MTLComputePipelineState>)state.offset_handle.get();
  if (state.block == nil || state.prefix == nil || state.offset == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
