#include "local.hpp"

namespace rund::kernel::reduction::fold {

FoldGraphValidationResult ValidateFoldGraphEdges(const FoldGraphView graph,
                                                 const std::span<u64> defined_markers,
                                                 const FoldPrimitiveSpec& primitive,
                                                 const u32 required_slots,
                                                 u32& final_slot) {
  u32 last_level = 0u;
  final_slot = graph.partition_fold_slots[graph.partition_count - 1u];
  for (u32 edge_index = 0u; edge_index < graph.reduction_edge_count; ++edge_index) {
    const FoldGraphEdge& edge = graph.reduction_edges[edge_index];
    if (edge.operation != graph.operation) {
      return FailFoldGraphValidation("fold_graph_primitive_mismatch", required_slots);
    }
    if (edge_index != 0u && edge.level < last_level) {
      return FailFoldGraphValidation("fold_graph_topological_order_required", required_slots);
    }
    last_level = edge.level;
    if (edge.left_slot >= required_slots ||
        edge.right_slot >= required_slots ||
        edge.output_slot >= required_slots) {
      return FailFoldGraphValidation("fold_graph_edge_out_of_range", required_slots);
    }
    if (defined_markers[edge.left_slot] == 0u) {
      return FailFoldGraphValidation("fold_graph_edge_input_unavailable", required_slots);
    }
    if (!edge.right_is_padding && defined_markers[edge.right_slot] == 0u) {
      return FailFoldGraphValidation("fold_graph_edge_input_unavailable", required_slots);
    }
    if (defined_markers[edge.output_slot] != 0u) {
      return FailFoldGraphValidation("fold_graph_edge_output_redefined", required_slots);
    }
    if (edge.right_is_padding &&
        (edge.padding_law != primitive.padding_law || edge.padding_value != primitive.padding_value)) {
      return FailFoldGraphValidation("fold_graph_padding_law_mismatch", required_slots);
    }
    if (!edge.right_is_padding && edge.padding_law != FoldPaddingLaw::None) {
      return FailFoldGraphValidation("fold_graph_padding_law_mismatch", required_slots);
    }
    defined_markers[edge.output_slot] =
        kFoldGraphValidationEdgeMarker | static_cast<u64>(edge_index);
    final_slot = edge.output_slot;
  }
  return FoldGraphValidationResult{.ok = true, .reason = "pass"};
}

} // namespace rund::kernel::reduction::fold
