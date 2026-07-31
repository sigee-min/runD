#pragma once

#include <accel/device.hpp>

#include "work.hpp"

namespace node_accel_contract::metal_stats {

[[nodiscard]] inline rund::kernel::BindingSet
BindingsFor(const NodeComputeOp &op, const rund::AccelDevice &pick, Work &work,
            std::array<rund::kernel::BufferSpan, 2u> &input_buffers,
            std::array<rund::kernel::u32, 4u> &staged) {
  rund::kernel::BindingSet bindings =
      rund::kernel::BindingSet::staged_outputs(staged.data(), staged.size());
  bindings.phase_id = Phase().phase_id;
  bindings.lane_count = 1u;
  bindings.op_hash_hi = op.map.op_hash_hi;
  bindings.op_hash_lo = op.map.op_hash_lo;
  bindings.api = pick.caps.api;
  bindings.scalar = rund::kernel::ComputeScalar::Lane32;
  bindings.input_bytes_per_tile = op.map.input_bytes_per_tile;
  bindings.output_bytes_per_tile = op.map.output_bytes_per_tile;
  bindings.param_bytes = op.map.param_bytes;
  bindings.metadata_bytes_per_tile = op.map.metadata_bytes_per_tile;
  bindings.input_buffers = input_buffers.data();
  bindings.input_buffer_count = op.map.input_buffer_count;
  bindings.param_data = work.params.data();
  bindings.param_data_bytes = sizeof(work.params);
  bindings.sequence_tiles = work.sequence_tiles.data();
  bindings.sequence_tile_count = work.sequence_tiles.size();
  return bindings;
}

} // namespace node_accel_contract::metal_stats
