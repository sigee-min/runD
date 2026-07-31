#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalPartitionCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalPartitionCommandState &state) {
  state.partition =
      static_cast<MetalPartitionEncodeResources *>(resources.get());
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state.partition == nullptr || state.partition->adapter != &adapter ||
      state.encoder == nil || !MetalPartitionBuffersReady(*state.partition)) {
    SetMetalLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  state.classify = (__bridge id<MTLComputePipelineState>)
                       state.partition->pipelines.classify.get();
  state.scatter = (__bridge id<MTLComputePipelineState>)
                      state.partition->pipelines.scatter.get();
  state.flags =
      (__bridge id<MTLBuffer>)state.partition->flags.device_buffer.get();
  state.values =
      (__bridge id<MTLBuffer>)state.partition->values.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.partition->output.device_buffer.get();
  state.flags_offset =
      static_cast<NSUInteger>(state.partition->flags.ref.offset_bytes);
  state.values_offset =
      static_cast<NSUInteger>(state.partition->values.ref.offset_bytes);
  state.output_offset =
      static_cast<NSUInteger>(state.partition->output.ref.offset_bytes);
  state.false_bits =
      (__bridge id<MTLBuffer>)state.partition->false_bits.buffer.get();
  state.false_offsets =
      (__bridge id<MTLBuffer>)state.partition->false_offsets.buffer.get();
  state.false_bits_offset =
      static_cast<NSUInteger>(state.partition->false_bits.offset);
  state.false_offsets_offset =
      static_cast<NSUInteger>(state.partition->false_offsets.offset);
  if (state.classify == nil || state.scatter == nil || state.flags == nil ||
      state.values == nil || state.output == nil || state.false_bits == nil ||
      state.false_offsets == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
