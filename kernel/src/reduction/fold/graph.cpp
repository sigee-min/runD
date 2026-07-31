#include "local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace rund::kernel {
namespace reduction::fold {

u32 FixedBinaryTreeEdgeCount(const u32 slot_count) {
  u32 edges = 0u;
  for (u32 active_count = slot_count; active_count > 1u; active_count = (active_count + 1u) / 2u) {
    edges += (active_count + 1u) / 2u;
  }
  return edges;
}

} // namespace reduction::fold

u32 FoldGraphFixedBinaryTreeEdgeCount(const u32 partition_count) {
  return reduction::fold::FixedBinaryTreeEdgeCount(partition_count);
}

u32 FoldGraphFixedBinaryTreeNodeCount(const u32 partition_count) {
  if (partition_count == 0u ||
      partition_count > std::numeric_limits<u32>::max() / 2u) {
    return 0u;
  }
  const u32 worker_and_ordered_nodes = partition_count * 2u;
  const u32 edge_count = reduction::fold::FixedBinaryTreeEdgeCount(partition_count);
  if (edge_count > std::numeric_limits<u32>::max() - worker_and_ordered_nodes) {
    return 0u;
  }
  return worker_and_ordered_nodes + edge_count;
}

u32 FoldGraphFixedBinaryTreeScratchSlotCount(const u32 partition_count) {
  return partition_count == 0u ? 0u : partition_count + reduction::fold::FixedBinaryTreeEdgeCount(partition_count);
}

u32 FoldGraphScratchSlotCount(const FoldGraphView graph) {
  u32 required = graph.scratch_slot_count;
  if (required == 0u) {
    required = graph.partition_count;
  }
  if (graph.partition_fold_slot_count != 0u && graph.partition_fold_slots == nullptr) {
    return 0u;
  }
  if (graph.node_count != 0u && graph.nodes == nullptr) {
    return 0u;
  }
  if (graph.reduction_edge_count != 0u && graph.reduction_edges == nullptr) {
    return 0u;
  }
  for (u32 slot_index = 0u; slot_index < graph.partition_fold_slot_count; ++slot_index) {
    required = std::max(required, graph.partition_fold_slots[slot_index] + 1u);
  }
  for (u32 edge_index = 0u; edge_index < graph.reduction_edge_count; ++edge_index) {
    const FoldGraphEdge& edge = graph.reduction_edges[edge_index];
    required = std::max(required, edge.left_slot + 1u);
    required = std::max(required, edge.right_slot + 1u);
    required = std::max(required, edge.output_slot + 1u);
  }
  return required;
}

FoldGraphValidationResult ValidateFoldGraph(const FoldGraphView graph) {
  const u32 required_slots = FoldGraphScratchSlotCount(graph);
  if (required_slots == 0u) {
    return reduction::fold::ValidateFoldGraphWithMarkers(graph, std::span<u64>{});
  }
  std::vector<u64> defined_markers(required_slots, 0u);
  return reduction::fold::ValidateFoldGraphWithMarkers(graph,
                                                       std::span<u64>(defined_markers.data(),
                                                                      defined_markers.size()));
}

} // namespace rund::kernel
