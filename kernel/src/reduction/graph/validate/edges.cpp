#include "local.hpp"

namespace rund::kernel::reduction::graph::validate_detail {

FoldGraphValidationResult ValidateFixedGraphReductionEdges(const FoldGraph& graph,
                                                           const FoldOperation operation,
                                                           const FoldPrimitiveSpec& primitive,
                                                           const u32 partition_count,
                                                           const u32 required_edges,
                                                           const u32 required_scratch_slots) {
  u32 edge_index = 0u;
  u32 level = 0u;
  u32 current_slot_base = 0u;
  u32 active_count = partition_count;
  u32 next_output_base = partition_count;
  u32 final_slot = partition_count - 1u;
  while (active_count > 1u) {
    const u32 output_count = (active_count + 1u) / 2u;
    for (u32 pair = 0u; pair < active_count; pair += 2u) {
      if (edge_index >= required_edges) {
        return FailBuiltGraphValidation("fold_graph_edge_missing", required_scratch_slots);
      }
      const bool has_right = pair + 1u < active_count;
      const u32 left_slot = current_slot_base + pair;
      const u32 right_slot = has_right ? current_slot_base + pair + 1u : current_slot_base + pair;
      const u32 output_slot = next_output_base + (pair / 2u);
      if (left_slot >= required_scratch_slots ||
          right_slot >= required_scratch_slots ||
          output_slot >= required_scratch_slots) {
        return FailBuiltGraphValidation("fold_graph_edge_out_of_range", required_scratch_slots);
      }
      const FoldGraphEdge& edge = graph.reduction_edges[edge_index];
      if (edge.level != level ||
          edge.left_slot != left_slot ||
          edge.right_slot != right_slot ||
          edge.output_slot != output_slot ||
          edge.operation != operation ||
          edge.right_is_padding == has_right) {
        return FailBuiltGraphValidation("fold_graph_edge_out_of_range", required_scratch_slots);
      }
      if (has_right && edge.padding_law != FoldPaddingLaw::None) {
        return FailBuiltGraphValidation("fold_graph_padding_law_mismatch", required_scratch_slots);
      }
      if (!has_right &&
          (edge.padding_law != primitive.padding_law ||
           edge.padding_value != primitive.padding_value)) {
        return FailBuiltGraphValidation("fold_graph_padding_law_mismatch", required_scratch_slots);
      }
      const FoldGraphNode& reduction_node = graph.nodes[partition_count * 2u + edge_index];
      if (reduction_node.kind != FoldGraphNodeKind::Reduction ||
          reduction_node.topological_index != partition_count * 2u + edge_index ||
          reduction_node.slot != edge.output_slot ||
          reduction_node.left_slot != edge.left_slot ||
          reduction_node.right_slot != edge.right_slot ||
          reduction_node.operation != operation ||
          reduction_node.value_domain != primitive.value_domain ||
          reduction_node.padding_law != edge.padding_law ||
          reduction_node.overflow_law != primitive.overflow_law ||
          reduction_node.right_is_padding != edge.right_is_padding) {
        return FailBuiltGraphValidation("fold_graph_node_mismatch", required_scratch_slots);
      }
      final_slot = output_slot;
      edge_index += 1u;
    }
    current_slot_base = next_output_base;
    next_output_base += output_count;
    active_count = output_count;
    level += 1u;
  }
  if (edge_index != required_edges || graph.final_slot != final_slot) {
    return FailBuiltGraphValidation("fold_graph_final_slot_mismatch", required_scratch_slots);
  }
  return FoldGraphValidationResult{
      .ok = true,
      .reason = "pass",
      .scratch_slot_count = required_scratch_slots,
      .final_slot = final_slot,
      .dag_validated = true,
      .slot_bounds_validated = true,
      .padding_law_validated = true,
  };
}

} // namespace rund::kernel::reduction::graph::validate_detail
