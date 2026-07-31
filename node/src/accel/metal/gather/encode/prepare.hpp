#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalGatherCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalGatherCommandState &state) {
  state.gather = static_cast<MetalGatherEncodeResources *>(resources.get());
  if (state.gather == nullptr || state.gather->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_gather_invalid");
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.control_pipeline = (__bridge id<MTLComputePipelineState>)
      state.gather->control_pipeline.get();
  state.gather_pipeline = (__bridge id<MTLComputePipelineState>)
      state.gather->gather_pipeline.get();
  state.values =
      (__bridge id<MTLBuffer>)state.gather->values.device_buffer.get();
  state.indices =
      (__bridge id<MTLBuffer>)state.gather->indices.device_buffer.get();
  state.logical_count =
      state.gather->plan.count_source ==
              rund::kernel::ComputeCountSource::Descriptor
          ? state.values
          : (__bridge id<MTLBuffer>)
                state.gather->logical_count.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.gather->output.device_buffer.get();
  state.status = (__bridge id<MTLBuffer>)state.gather->status.buffer.get();
  state.indirect =
      (__bridge id<MTLBuffer>)state.gather->indirect.buffer.get();
  if (state.encoder == nil || state.control_pipeline == nil ||
      state.gather_pipeline == nil || state.values == nil ||
      state.indices == nil || state.logical_count == nil ||
      state.output == nil || state.status == nil || state.indirect == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
