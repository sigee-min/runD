#include "local.hpp"

namespace rund::kernel::reduction::fold {
namespace {

bool ReductionNodeMatchesEdge(const FoldGraphNode& node, const FoldGraphEdge& edge) {
  return node.slot == edge.output_slot &&
         node.left_slot == edge.left_slot &&
         node.right_slot == edge.right_slot &&
         node.operation == edge.operation &&
         node.padding_law == edge.padding_law &&
         node.right_is_padding == edge.right_is_padding;
}

bool IsPartitionMarker(const u64 marker) {
  return marker == kFoldGraphValidationPartitionMarker;
}

bool IsEdgeMarker(const u64 marker) {
  return (marker & kFoldGraphValidationMarkerTagMask) == kFoldGraphValidationEdgeMarker;
}

bool PartitionNodeValid(const std::span<const u64> defined_markers, const FoldGraphNode& node) {
  return IsPartitionMarker(defined_markers[node.slot]) &&
         node.left_slot == node.slot &&
         node.right_slot == node.slot &&
         node.padding_law == FoldPaddingLaw::None &&
         !node.right_is_padding;
}

} // namespace

FoldGraphValidationResult ValidateFoldGraphNodes(const FoldGraphView graph,
                                                 const std::span<const u64> defined_markers,
                                                 const FoldPrimitiveSpec& primitive,
                                                 const u32 required_slots) {
  if (defined_markers.size() < static_cast<std::size_t>(required_slots)) {
    return FailFoldGraphValidation("fold_graph_scratch_capacity_exceeded", required_slots);
  }
  for (u32 node_index = 0u; node_index < graph.node_count; ++node_index) {
    const FoldGraphNode& node = graph.nodes[node_index];
    if (node.topological_index != node_index || node.slot >= required_slots ||
        node.operation != graph.operation || node.value_domain != primitive.value_domain ||
        node.overflow_law != primitive.overflow_law) {
      return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
    }
    if (node.right_is_padding &&
        (node.padding_law != primitive.padding_law || node.right_slot >= required_slots)) {
      return FailFoldGraphValidation("fold_graph_padding_law_mismatch", required_slots);
    }
    switch (node.kind) {
      case FoldGraphNodeKind::WorkerLocalPartial:
      case FoldGraphNodeKind::GlobalOrderedSlot:
        if (!PartitionNodeValid(defined_markers, node)) {
          return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
        }
        break;
      case FoldGraphNodeKind::Reduction:
        if (!IsEdgeMarker(defined_markers[node.slot])) {
          return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
        }
        {
          const u64 edge_index =
              defined_markers[node.slot] & kFoldGraphValidationMarkerPayloadMask;
          if (edge_index >= static_cast<u64>(graph.reduction_edge_count)) {
            return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
          }
          if (!ReductionNodeMatchesEdge(node,
                                        graph.reduction_edges[static_cast<std::size_t>(edge_index)])) {
            return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
          }
        }
        break;
      default:
        return FailFoldGraphValidation("fold_graph_node_mismatch", required_slots);
    }
  }
  return FoldGraphValidationResult{.ok = true, .reason = "pass"};
}

} // namespace rund::kernel::reduction::fold
