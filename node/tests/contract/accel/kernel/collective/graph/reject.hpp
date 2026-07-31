#pragma once

#include <accel/graph/node.hpp>

#include "parity.hpp"

namespace node_accel_contract::collective::graph_case {

[[nodiscard]] inline bool InvalidCollectiveNodesReject(const State &state) {
  rund::AccelGraphNode mismatched = state.nodes[0];
  mismatched.primitive_hash_lo ^= 1u;
  if (!SingleNodeCompileReason(state.context, state.graph, mismatched,
                               "accel_kernel_graph_invalid")) {
    return false;
  }

  rund::AccelGraphNode invalid_key_bits = state.nodes[0];
  invalid_key_bits.sort.key_bits = 16u;
  if (!SingleNodeCompileReason(state.context, state.graph, invalid_key_bits,
                               "accel_kernel_graph_invalid")) {
    return false;
  }

  rund::AccelGraphNode mixed_collective = state.nodes[0];
  mixed_collective.ir = &state.op.ir();
  if (!SingleNodeCompileReason(state.context, state.graph, mixed_collective,
                               "accel_kernel_graph_invalid")) {
    return false;
  }

  const rund::AccelGraphNode mixed_map{
      .ir = &state.op.ir(),
      .buffers = state.refs.data(),
      .buffer_count = state.refs.size(),
      .kind = rund::kernel::NodeKind::Map,
      .primitive_hash_hi = 0x9999aaaabbbbccccu,
      .element_count = state.input.count,
  };
  return SingleNodeCompileReason(state.context, state.graph, mixed_map,
                                 "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::collective::graph_case
