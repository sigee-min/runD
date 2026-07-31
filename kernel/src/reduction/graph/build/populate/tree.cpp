#include "../local.hpp"

namespace rund::kernel::reduction::graph::build_detail {

void AppendReductionTree(FoldGraph& graph, const BuildPlan& plan) {
  u32 edge_index = 0u;
  u32 level = 0u;
  u32 current_slot_base = 0u;
  u32 active_count = plan.partition_count;
  u32 next_output_base = plan.partition_count;
  while (active_count > 1u) {
    const u32 output_count = (active_count + 1u) / 2u;
    for (u32 pair = 0u; pair < active_count; pair += 2u) {
      const bool has_right = pair + 1u < active_count;
      const FoldGraphEdge edge{
          .level = level,
          .left_slot = current_slot_base + pair,
          .right_slot = has_right ? current_slot_base + pair + 1u : current_slot_base + pair,
          .output_slot = next_output_base + (pair / 2u),
          .operation = plan.operation,
          .padding_law = has_right ? FoldPaddingLaw::None : plan.primitive.padding_law,
          .padding_value = has_right ? 0u : plan.primitive.padding_value,
          .right_is_padding = !has_right,
      };
      graph.reduction_edges[edge_index++] = edge;
      graph.nodes.push_back(FoldGraphNode{
          .kind = FoldGraphNodeKind::Reduction,
          .topological_index = static_cast<u32>(graph.nodes.size()),
          .slot = edge.output_slot,
          .left_slot = edge.left_slot,
          .right_slot = edge.right_slot,
          .operation = plan.operation,
          .value_domain = plan.primitive.value_domain,
          .padding_law = edge.padding_law,
          .overflow_law = plan.primitive.overflow_law,
          .right_is_padding = edge.right_is_padding,
      });
      graph.final_slot = edge.output_slot;
    }
    current_slot_base = next_output_base;
    next_output_base += output_count;
    active_count = output_count;
    level += 1u;
  }
}

} // namespace rund::kernel::reduction::graph::build_detail
