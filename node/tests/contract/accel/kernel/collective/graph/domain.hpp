#pragma once

#include <accel/graph/node.hpp>

#include "state.hpp"

namespace node_accel_contract::collective::graph_case {

[[nodiscard]] inline bool
DomainGraphKindSlotRejectsAtCompile(const State &state,
                                    const rund::kernel::u8 slot) {
  const rund::AccelGraphNode node{
      .buffers = state.refs.data(),
      .buffer_count = state.refs.size(),
      .kind = static_cast<rund::kernel::NodeKind>(slot),
      .primitive_hash_hi = 0x1111222233334444u,
      .primitive_hash_lo = 0x5555666677778888u,
      .element_count = state.input.count,
  };
  return SingleNodeCompileReason(state.context, state.graph, node,
                                 "accel_kernel_graph_invalid");
}

[[nodiscard]] inline bool
ReservedGraphKindSlotsRejectAtCompile(const State &state) {
  return DomainGraphKindSlotRejectsAtCompile(state, 5u) &&
         DomainGraphKindSlotRejectsAtCompile(state, 6u) &&
         DomainGraphKindSlotRejectsAtCompile(state, 7u);
}

} // namespace node_accel_contract::collective::graph_case
