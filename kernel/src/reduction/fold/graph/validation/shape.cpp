#include "local.hpp"

namespace rund::kernel::reduction::fold {

FoldGraphValidationResult ValidateFoldGraphShape(const FoldGraphView graph) {
  if (!graph.primitive_standardized || !IsSupportedFoldOperation(graph.operation)) {
    return FailFoldGraphValidation("unsupported_fold_operation");
  }
  if (FoldOperationAllowsFloatingPoint(graph.operation)) {
    if (!graph.strict_floating_point) {
      return FailFoldGraphValidation("floating_point_fold_forbidden");
    }
    if (!graph.strict_float_reduction.valid) {
      return FailFoldGraphValidation("floating_point_law_required");
    }
    if (!StrictFloatReductionValid(graph.operation, graph.strict_float_reduction)) {
      return FailFoldGraphValidation("floating_point_law_mismatch");
    }
  } else if (graph.floating_point_allowed || graph.strict_floating_point) {
    return FailFoldGraphValidation("floating_point_fold_forbidden");
  }
  if (!graph.fixed_topological_order) {
    return FailFoldGraphValidation("fold_graph_topological_order_required");
  }
  if (graph.partition_count == 0u ||
      graph.partition_fold_slots == nullptr ||
      graph.partition_fold_slot_count != graph.partition_count) {
    return FailFoldGraphValidation("fold_graph_partition_mismatch");
  }
  if (graph.reduction_edge_count != 0u && graph.reduction_edges == nullptr) {
    return FailFoldGraphValidation("fold_graph_edge_missing");
  }
  if (graph.node_count != 0u && graph.nodes == nullptr) {
    return FailFoldGraphValidation("fold_graph_node_missing");
  }
  return FoldGraphValidationResult{.ok = true, .reason = "pass"};
}

} // namespace rund::kernel::reduction::fold
