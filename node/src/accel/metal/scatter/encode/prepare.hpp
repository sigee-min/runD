#pragma once

#include <accel/check.hpp>

#include "state.hpp"
#include "status.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalScatterCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalScatterCommandState &state) {
  state.scatter = static_cast<MetalScatterEncodeResources *>(resources.get());
  if (state.scatter == nullptr || state.scatter->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_scatter_invalid");
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.pipeline =
      (__bridge id<MTLComputePipelineState>)state.scatter->pipeline.get();
  state.values =
      (__bridge id<MTLBuffer>)state.scatter->values.device_buffer.get();
  state.indices =
      (__bridge id<MTLBuffer>)state.scatter->indices.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.scatter->output.device_buffer.get();
  const rund::AccelCheck reset =
      ResetMetalScatterStatus(adapter, *state.scatter);
  if (!reset.ok) {
    return reset;
  }
  state.status = (__bridge id<MTLBuffer>)state.scatter->status.buffer.get();
  if (state.encoder == nil || state.pipeline == nil || state.values == nil ||
      state.indices == nil || state.output == nil || state.status == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
