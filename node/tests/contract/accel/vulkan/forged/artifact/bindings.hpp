#pragma once

#include "lowering.hpp"

namespace node_accel_contract::vulkan {

struct ForgedArtifactBindings {
  std::array<rund::kernel::BufferSpan, 2u> inputs{};
  rund::kernel::BindingSet bindings{};
  rund::kernel::ComputeDispatchWindow window{};
};

inline void PrepareForgedArtifactBindings(
    ForgedArtifactBindings& out,
    ForgedArtifactWork& work,
    const ForgedArtifactLowering& lowering) {
  out.inputs = {
      rund::kernel::BufferSpan::contiguous(work.lhs.data(),
                                                  work.lhs.size()),
      rund::kernel::BufferSpan::contiguous(work.rhs.data(),
                                                  work.rhs.size()),
  };
  out.bindings =
      rund::kernel::BindingSet::staged_outputs(work.output.data(),
                                                      work.output.size());
  out.bindings.phase_id = Phase().phase_id;
  out.bindings.lane_count = 1u;
  out.bindings.op_hash_hi = lowering.map.op_hash_hi;
  out.bindings.op_hash_lo = lowering.map.op_hash_lo;
  out.bindings.api = lowering.plan.api;
  out.bindings.scalar = lowering.plan.scalar;
  out.bindings.input_bytes_per_tile = lowering.plan.input_bytes_per_tile;
  out.bindings.output_bytes_per_tile = lowering.plan.output_bytes_per_tile;
  out.bindings.param_bytes = lowering.plan.param_bytes;
  out.bindings.metadata_bytes_per_tile = lowering.plan.metadata_bytes_per_tile;
  out.bindings.input_buffers = out.inputs.data();
  out.bindings.input_buffer_count = out.inputs.size();
  out.bindings.param_data = &work.scale;
  out.bindings.param_data_bytes = sizeof(work.scale);
  out.bindings.sequence_tiles = work.sequence_tiles.data();
  out.bindings.sequence_tile_count = work.sequence_tiles.size();
  out.window = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = lowering.plan.dispatch_window_tiles,
  };
}

}  // namespace node_accel_contract::vulkan
