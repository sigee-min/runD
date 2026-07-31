#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>

#include <kernel/program/compute/limit.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

bool GraphShapeCanBeWalked(const rund::AccelGraph &graph) noexcept {
  if (graph.output_count > rund::kernel::kMaxGraphOutputCount ||
      (graph.output_count != 0u && graph.outputs == nullptr)) {
    return false;
  }
  if (graph.node_count == 0u) {
    return true;
  }
  return graph.nodes != nullptr &&
         graph.node_count <= rund::kernel::kMaxGraphNodeCount;
}

bool NodeBuffersCanBeWalked(const rund::AccelGraphNode &node) noexcept {
  if (node.buffer_count == 0u) {
    return true;
  }
  return node.buffers != nullptr &&
         node.buffer_count <= rund::kernel::kMaxGraphBuffersPerNode;
}

void ReserveGraphCompileState(GraphCompileState &state,
                              const std::uint64_t node_count) {
  const std::size_t count = static_cast<std::size_t>(node_count);
  state.buffer_storage.reserve(count);
  state.node_storage.reserve(count);
  state.compile_nodes.reserve(count);
  state.fusion_nodes.reserve(count);
  state.required_barriers.reserve(count);
}

bool ReserveExplicitGraphLogicalIds(const rund::AccelGraph &graph,
                                    GraphCompileState &state) {
  for (std::uint64_t node_index = 0u; node_index < graph.node_count;
       ++node_index) {
    const rund::AccelGraphNode &node = graph.nodes[node_index];
    if (!NodeBuffersCanBeWalked(node)) {
      return false;
    }
    for (std::uint64_t buffer_index = 0u; buffer_index < node.buffer_count;
         ++buffer_index) {
      const std::uint64_t logical_id = node.buffers[buffer_index].logical_id;
      if (logical_id != 0u) {
        state.explicit_logical_ids.insert(logical_id);
      }
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
