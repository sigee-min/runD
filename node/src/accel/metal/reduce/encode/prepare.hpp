#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalReduceCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalReduceCommandState &state) {
  state.reduce = static_cast<MetalReduceEncodeResources *>(resources.get());
  if (state.reduce == nullptr || state.reduce->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_reduce_invalid");
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.pipeline =
      (__bridge id<MTLComputePipelineState>)state.reduce->pipeline.get();
  state.input = (__bridge id<MTLBuffer>)state.reduce->input.device_buffer.get();
  state.partial = (__bridge id<MTLBuffer>)state.reduce->partial.buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.reduce->output.device_buffer.get();
  state.status = (__bridge id<MTLBuffer>)state.reduce->status.buffer.get();
  state.logical_count =
      (__bridge id<MTLBuffer>)state.reduce->logical_count.device_buffer.get();
  state.input_offset =
      static_cast<NSUInteger>(state.reduce->input.ref.offset_bytes);
  state.partial_offset =
      static_cast<NSUInteger>(state.reduce->partial.offset);
  state.output_offset =
      static_cast<NSUInteger>(state.reduce->output.ref.offset_bytes);
  state.logical_count_offset =
      static_cast<NSUInteger>(state.reduce->logical_count.ref.offset_bytes);
  if (state.encoder == nil || state.pipeline == nil || state.input == nil ||
      state.partial == nil || state.output == nil || state.status == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [state.encoder setComputePipelineState:state.pipeline];
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
