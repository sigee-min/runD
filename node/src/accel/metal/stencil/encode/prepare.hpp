#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalStencilCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalStencilCommandState &state) {
  state.stencil = static_cast<MetalStencilEncodeResources *>(resources.get());
  if (state.stencil == nullptr || state.stencil->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.pipeline =
      (__bridge id<MTLComputePipelineState>)state.stencil->pipeline.get();
  state.input =
      (__bridge id<MTLBuffer>)state.stencil->input.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.stencil->output.device_buffer.get();
  if (state.encoder == nil || state.pipeline == nil || state.input == nil ||
      state.output == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
