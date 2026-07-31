#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalCompactEncodeState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalCompactEncodeState &state) {
  state.compact = static_cast<MetalCompactEncodeResources *>(resources.get());
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state.compact == nullptr || state.compact->adapter != &adapter ||
      state.encoder == nil) {
    SetMetalLastError(adapter, "compute_compact_invalid");
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  state.scatter = (__bridge id<MTLComputePipelineState>)
                      state.compact->pipelines.scatter.get();
  state.count_blocks = (__bridge id<MTLComputePipelineState>)
                           state.compact->pipelines.count_blocks.get();
  state.scatter_blocks = (__bridge id<MTLComputePipelineState>)
                             state.compact->pipelines.scatter_blocks.get();
  state.status_pipeline = (__bridge id<MTLComputePipelineState>)
                              state.compact->pipelines.status.get();
  state.status_required = state.compact->plan.status_bytes != 0u;
  if ((state.compact->block_offset_path &&
       (state.count_blocks == nil || state.scatter_blocks == nil)) ||
      (!state.compact->block_offset_path &&
       (state.scatter == nil ||
        (state.status_required && state.status_pipeline == nil)))) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  state.flags =
      (__bridge id<MTLBuffer>)state.compact->flags.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.compact->output.device_buffer.get();
  state.flags_offset =
      static_cast<NSUInteger>(state.compact->flags.ref.offset_bytes);
  state.output_offset =
      static_cast<NSUInteger>(state.compact->output.ref.offset_bytes);
  state.offsets = (__bridge id<MTLBuffer>)state.compact->offsets.buffer.get();
  state.flag_bits =
      (__bridge id<MTLBuffer>)state.compact->flag_bits.buffer.get();
  state.block_counts =
      (__bridge id<MTLBuffer>)state.compact->block_counts.buffer.get();
  state.block_offsets =
      (__bridge id<MTLBuffer>)state.compact->block_offsets.buffer.get();
  state.status = (__bridge id<MTLBuffer>)state.compact->status.buffer.get();
  state.offsets_offset =
      static_cast<NSUInteger>(state.compact->offsets.offset);
  state.flag_bits_offset =
      static_cast<NSUInteger>(state.compact->flag_bits.offset);
  state.block_counts_offset =
      static_cast<NSUInteger>(state.compact->block_counts.offset);
  state.block_offsets_offset =
      static_cast<NSUInteger>(state.compact->block_offsets.offset);
  state.scan_totals_offset =
      static_cast<NSUInteger>(state.compact->scan_totals.offset);
  state.params = CompactParams{
      state.compact->plan.element_count,
      state.compact->plan.output_capacity,
      state.compact->scan_plan.pass_count == 2u
          ? static_cast<rund::kernel::u32>(state.compact->scan_plan.block_size)
          : 0u,
      0u,
  };
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
