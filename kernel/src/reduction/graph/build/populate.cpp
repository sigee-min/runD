#include "local.hpp"

namespace rund::kernel::reduction::graph::build_detail {

void PopulateFixedBinaryTreeGraph(FoldGraph& graph, const BuildPlan& plan) {
  graph.operation = plan.operation;
  graph.value_domain = plan.primitive.value_domain;
  graph.strict_float_reduction = plan.strict_float_reduction;
  graph.partition_count = plan.partition_count;
  graph.worker_local_slot_count = plan.partition_count;
  graph.global_ordered_slot_count = plan.partition_count;
  graph.scratch_slot_count = plan.required_scratch_slots;
  graph.final_slot = plan.partition_count - 1u;
  graph.fixed_binary_tree = true;
  graph.fixed_topological_order = true;
  graph.dag_validated = false;
  graph.slot_bounds_validated = false;
  graph.padding_law_validated = false;
  graph.primitive_standardized = plan.primitive.supported;
  graph.floating_point_allowed = false;
  graph.strict_floating_point = plan.primitive.requires_strict_floating_point;
  graph.partition_fold_slots.resize(plan.partition_count);
  graph.nodes.clear();
  graph.nodes.reserve(plan.required_nodes);
  AppendPartitionSlots(graph, plan);
  graph.reduction_edges.clear();
  graph.reduction_edges.resize(plan.required_edges);
  AppendReductionTree(graph, plan);
}

} // namespace rund::kernel::reduction::graph::build_detail
