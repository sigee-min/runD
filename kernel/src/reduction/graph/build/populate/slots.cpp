#include "../local.hpp"

namespace rund::kernel::reduction::graph::build_detail {

void AppendPartitionSlots(FoldGraph& graph, const BuildPlan& plan) {
  for (u32 partition = 0u; partition < plan.partition_count; ++partition) {
    graph.partition_fold_slots[partition] = partition;
    graph.nodes.push_back(FoldGraphNode{
        .kind = FoldGraphNodeKind::WorkerLocalPartial,
        .topological_index = static_cast<u32>(graph.nodes.size()),
        .slot = partition,
        .left_slot = partition,
        .right_slot = partition,
        .operation = plan.operation,
        .value_domain = plan.primitive.value_domain,
        .padding_law = FoldPaddingLaw::None,
        .overflow_law = plan.primitive.overflow_law,
        .right_is_padding = false,
    });
    graph.nodes.push_back(FoldGraphNode{
        .kind = FoldGraphNodeKind::GlobalOrderedSlot,
        .topological_index = static_cast<u32>(graph.nodes.size()),
        .slot = partition,
        .left_slot = partition,
        .right_slot = partition,
        .operation = plan.operation,
        .value_domain = plan.primitive.value_domain,
        .padding_law = FoldPaddingLaw::None,
        .overflow_law = plan.primitive.overflow_law,
        .right_is_padding = false,
    });
  }
}

} // namespace rund::kernel::reduction::graph::build_detail
