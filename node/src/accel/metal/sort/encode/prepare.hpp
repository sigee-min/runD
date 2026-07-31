#pragma once

#include <accel/check.hpp>

#include "state.hpp"
#include "../../../sort/block/metal.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalSortEncodeState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalSortEncodeState &state) {
  state.sort = static_cast<MetalSortEncodeResources *>(resources.get());
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state.sort == nullptr || state.sort->adapter != &adapter ||
      state.encoder == nil || state.sort->block_count == 0u ||
      state.sort->plan.radix_pass_count == 0u ||
      state.sort->plan.radix_pass_count > 8u) {
    SetMetalLastError(adapter, "compute_sort_invalid");
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  state.dispatch = (__bridge id<MTLComputePipelineState>)
                       state.sort->pipelines.dispatch.get();
  state.histogram = (__bridge id<MTLComputePipelineState>)
                        state.sort->pipelines.histogram.get();
  state.prefix =
      (__bridge id<MTLComputePipelineState>)state.sort->pipelines.prefix.get();
  state.base =
      (__bridge id<MTLComputePipelineState>)state.sort->pipelines.base.get();
  state.scatter =
      (__bridge id<MTLComputePipelineState>)state.sort->pipelines.scatter.get();
  if (state.dispatch == nil || state.histogram == nil || state.prefix == nil ||
      state.base == nil || state.scatter == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  state.input_keys =
      (__bridge id<MTLBuffer>)state.sort->read_keys.device_buffer.get();
  if (state.sort->plan.value != rund::kernel::SortValue::IdentityU32) {
    state.input_values =
        (__bridge id<MTLBuffer>)state.sort->read_values.device_buffer.get();
  }
  state.output_keys =
      (__bridge id<MTLBuffer>)state.sort->write_keys.device_buffer.get();
  state.output_values =
      (__bridge id<MTLBuffer>)state.sort->write_values.device_buffer.get();
  state.temp_keys = (__bridge id<MTLBuffer>)state.sort->temp_keys.buffer.get();
  state.temp_values =
      (__bridge id<MTLBuffer>)state.sort->temp_values.buffer.get();
  state.block_counts =
      (__bridge id<MTLBuffer>)state.sort->block_counts.buffer.get();
  state.block_offsets =
      (__bridge id<MTLBuffer>)state.sort->block_offsets.buffer.get();
  state.bucket_offsets =
      (__bridge id<MTLBuffer>)state.sort->bucket_offsets.buffer.get();
  state.dispatch_args =
      (__bridge id<MTLBuffer>)state.sort->dispatch_args.buffer.get();
  state.status = (__bridge id<MTLBuffer>)state.sort->status.buffer.get();
  state.logical_count = state.sort->logical_count.device_buffer == nullptr
                            ? state.input_keys
                            : (__bridge id<MTLBuffer>)
                                  state.sort->logical_count.device_buffer.get();
  state.input_keys_offset =
      static_cast<NSUInteger>(state.sort->read_keys.ref.offset_bytes);
  state.input_values_offset =
      static_cast<NSUInteger>(state.sort->read_values.ref.offset_bytes);
  state.output_keys_offset =
      static_cast<NSUInteger>(state.sort->write_keys.ref.offset_bytes);
  state.output_values_offset =
      static_cast<NSUInteger>(state.sort->write_values.ref.offset_bytes);
  state.temp_keys_offset =
      static_cast<NSUInteger>(state.sort->temp_keys.offset);
  state.temp_values_offset =
      static_cast<NSUInteger>(state.sort->temp_values.offset);
  state.block_counts_offset =
      static_cast<NSUInteger>(state.sort->block_counts.offset);
  state.block_offsets_offset =
      static_cast<NSUInteger>(state.sort->block_offsets.offset);
  state.bucket_offsets_offset =
      static_cast<NSUInteger>(state.sort->bucket_offsets.offset);
  state.logical_count_offset =
      static_cast<NSUInteger>(state.sort->logical_count.ref.offset_bytes);
  state.groups = MTLSizeMake(state.sort->block_count, 1u, 1u);
  state.threads = MTLSizeMake(kMetalSortThreadCount, 1u, 1u);
  state.bounded = state.sort->plan.count_source !=
                  rund::kernel::ComputeCountSource::Descriptor;
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
