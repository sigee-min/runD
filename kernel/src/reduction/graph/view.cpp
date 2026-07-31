#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel {

FoldGraphView ViewFoldGraph(const FoldGraph& graph) {
  const FoldGraphProgram program{
      .operation = graph.operation,
      .value_domain = graph.value_domain,
      .strict_float_reduction = graph.strict_float_reduction,
      .partition_count = graph.partition_count,
      .worker_local_slot_count = graph.worker_local_slot_count,
      .global_ordered_slot_count = graph.global_ordered_slot_count,
      .scratch_slot_count = graph.scratch_slot_count,
      .final_slot = graph.final_slot,
      .nodes = graph.nodes.data(),
      .node_count = static_cast<u32>(graph.nodes.size()),
      .edges = graph.reduction_edges.data(),
      .edge_count = static_cast<u32>(graph.reduction_edges.size()),
      .fixed_topological_order = graph.fixed_topological_order,
      .dag_validated = graph.dag_validated,
      .slot_bounds_validated = graph.slot_bounds_validated,
      .padding_law_validated = graph.padding_law_validated,
      .primitive_standardized = graph.primitive_standardized,
      .strict_floating_point = graph.strict_floating_point,
  };
  return FoldGraphView{
      .operation = graph.operation,
      .value_domain = graph.value_domain,
      .strict_float_reduction = graph.strict_float_reduction,
      .partition_count = graph.partition_count,
      .worker_local_slot_count = graph.worker_local_slot_count,
      .global_ordered_slot_count = graph.global_ordered_slot_count,
      .scratch_slot_count = graph.scratch_slot_count,
      .final_slot = graph.final_slot,
      .fixed_binary_tree = graph.fixed_binary_tree,
      .fixed_topological_order = graph.fixed_topological_order,
      .dag_validated = graph.dag_validated,
      .slot_bounds_validated = graph.slot_bounds_validated,
      .padding_law_validated = graph.padding_law_validated,
      .primitive_standardized = graph.primitive_standardized,
      .floating_point_allowed = graph.floating_point_allowed,
      .strict_floating_point = graph.strict_floating_point,
      .partition_fold_slots = graph.partition_fold_slots.data(),
      .partition_fold_slot_count = static_cast<u32>(graph.partition_fold_slots.size()),
      .nodes = graph.nodes.data(),
      .node_count = static_cast<u32>(graph.nodes.size()),
      .reduction_edges = graph.reduction_edges.data(),
      .reduction_edge_count = static_cast<u32>(graph.reduction_edges.size()),
      .program = program,
  };
}

} // namespace rund::kernel
