#pragma once

#include "work.hpp"

namespace node_accel_contract::vulkan_stats {

[[nodiscard]] inline rund::kernel::BindingSet BindingsFor(
    const rund::kernel::ComputeMap& map,
    Work& work,
    std::array<rund::kernel::BufferSpan, 2u>& input_buffers,
    std::array<TileValue, 4u>& output) {
  rund::kernel::BindingSet bindings =
      rund::kernel::BindingSet::staged_outputs(output.data(),
                                                      output.size());
  bindings.phase_id = Phase().phase_id;
  bindings.lane_count = 1u;
  bindings.op_hash_hi = map.op_hash_hi;
  bindings.op_hash_lo = map.op_hash_lo;
  bindings.api = map.api;
  bindings.scalar = rund::kernel::ComputeScalar::Lane32;
  bindings.input_bytes_per_tile = map.input_bytes_per_tile;
  bindings.output_bytes_per_tile = map.output_bytes_per_tile;
  bindings.param_bytes = map.param_bytes;
  bindings.metadata_bytes_per_tile = map.metadata_bytes_per_tile;
  bindings.input_buffers = input_buffers.data();
  bindings.input_buffer_count = map.input_buffer_count;
  bindings.param_data = work.params.data();
  bindings.param_data_bytes = sizeof(work.params);
  bindings.sequence_tiles = work.sequence_tiles.data();
  bindings.sequence_tile_count = work.sequence_tiles.size();
  return bindings;
}

}  // namespace node_accel_contract::vulkan_stats
