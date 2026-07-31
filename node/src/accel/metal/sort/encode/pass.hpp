#pragma once

#include <accel/check.hpp>

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline MetalSortPassBuffers
SelectMetalSortPassBuffers(const MetalSortEncodeState &state,
                           const rund::kernel::u32 pass) {
  MetalSortPassBuffers buffers{};
  buffers.source_keys = state.input_keys;
  buffers.source_keys_offset = state.input_keys_offset;
  buffers.source_values =
      state.input_values == nil ? state.temp_values : state.input_values;
  buffers.source_values_offset =
      state.input_values == nil ? state.temp_values_offset
                                : state.input_values_offset;
  if (pass != 0u) {
    buffers.source_keys =
        (pass % 2u) == 1u ? state.temp_keys : state.output_keys;
    buffers.source_keys_offset =
        (pass % 2u) == 1u ? state.temp_keys_offset : state.output_keys_offset;
    buffers.source_values =
        (pass % 2u) == 1u ? state.temp_values : state.output_values;
    buffers.source_values_offset =
        (pass % 2u) == 1u ? state.temp_values_offset
                          : state.output_values_offset;
  }
  buffers.target_keys = (pass % 2u) == 0u ? state.temp_keys : state.output_keys;
  buffers.target_keys_offset =
      (pass % 2u) == 0u ? state.temp_keys_offset : state.output_keys_offset;
  buffers.target_values =
      (pass % 2u) == 0u ? state.temp_values : state.output_values;
  buffers.target_values_offset =
      (pass % 2u) == 0u ? state.temp_values_offset
                        : state.output_values_offset;
  return buffers;
}

inline void DispatchMetalSortBlocks(const MetalSortEncodeState &state) {
  if (state.bounded) {
    [state.encoder dispatchThreadgroupsWithIndirectBuffer:state.dispatch_args
                                     indirectBufferOffset:0u
                                    threadsPerThreadgroup:state.threads];
    return;
  }
  [state.encoder dispatchThreadgroups:state.groups
                threadsPerThreadgroup:state.threads];
}

inline void EncodeMetalSortPass(const MetalSortEncodeState &state,
                                const rund::kernel::u32 pass) {
  const MetalSortPassBuffers buffers = SelectMetalSortPassBuffers(state, pass);
  const SortParams params = MetalSortParams(*state.sort, pass);
  [state.encoder setComputePipelineState:state.histogram];
  [state.encoder setBuffer:buffers.source_keys
                    offset:buffers.source_keys_offset
                   atIndex:0u];
  [state.encoder setBuffer:state.block_counts
                    offset:state.block_counts_offset
                   atIndex:1u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:2u];
  [state.encoder setBuffer:state.logical_count
                    offset:state.logical_count_offset
                   atIndex:3u];
  DispatchMetalSortBlocks(state);
  [state.encoder setComputePipelineState:state.prefix];
  [state.encoder setBuffer:state.block_counts
                    offset:state.block_counts_offset
                   atIndex:0u];
  [state.encoder setBuffer:state.block_offsets
                    offset:state.block_offsets_offset
                   atIndex:1u];
  [state.encoder setBuffer:state.bucket_offsets
                    offset:state.bucket_offsets_offset
                   atIndex:2u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:3u];
  [state.encoder setBuffer:state.logical_count
                    offset:state.logical_count_offset
                   atIndex:4u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(kSortBucketCount, 1u,
                                                  1u)
                threadsPerThreadgroup:state.threads];
  [state.encoder setComputePipelineState:state.base];
  [state.encoder setBuffer:state.bucket_offsets
                    offset:state.bucket_offsets_offset
                   atIndex:0u];
  [state.encoder setBuffer:state.bucket_offsets
                    offset:state.bucket_offsets_offset
                   atIndex:1u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:2u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:state.threads];
  [state.encoder setComputePipelineState:state.scatter];
  [state.encoder setBuffer:buffers.source_keys
                    offset:buffers.source_keys_offset
                   atIndex:0u];
  [state.encoder setBuffer:buffers.source_values
                    offset:buffers.source_values_offset
                   atIndex:1u];
  [state.encoder setBuffer:buffers.target_keys
                    offset:buffers.target_keys_offset
                   atIndex:2u];
  [state.encoder setBuffer:buffers.target_values
                    offset:buffers.target_values_offset
                   atIndex:3u];
  [state.encoder setBuffer:state.block_offsets
                    offset:state.block_offsets_offset
                   atIndex:4u];
  [state.encoder setBuffer:state.bucket_offsets
                    offset:state.bucket_offsets_offset
                   atIndex:5u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:6u];
  [state.encoder setBuffer:state.logical_count
                    offset:state.logical_count_offset
                   atIndex:7u];
  DispatchMetalSortBlocks(state);
}
#endif

} // namespace rund::node::accel::detail
