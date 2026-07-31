#include "local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <algorithm>
#include <chrono>

namespace rund::kernel {

FoldResult FoldGraphReduce(const FoldGraphView graph,
                           const std::span<const u64> partition_values,
                           const std::span<u64> scratch_slots) {
  const auto start = std::chrono::steady_clock::now();
  const auto fail = [&](const char* const reason) {
    const auto end = std::chrono::steady_clock::now();
    return FoldResult{
        .ok = false,
        .reason = reason,
        .slot_count = graph.partition_count,
        .operation = graph.operation,
        .fold_cost_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        .fold_cost_measured = true,
    };
  };
  if (partition_values.size() != static_cast<std::size_t>(graph.partition_count)) {
    return fail("fold_graph_partition_mismatch");
  }
  const FoldGraphValidationResult validation = reduction::fold::ValidateFoldGraphWithMarkers(graph, scratch_slots);
  if (!validation.ok) {
    return fail(validation.reason);
  }
  const u32 required_slots = validation.scratch_slot_count;
  if (scratch_slots.size() < static_cast<std::size_t>(required_slots)) {
    return fail("fold_graph_scratch_capacity_exceeded");
  }
  std::fill(scratch_slots.begin(), scratch_slots.end(), reduction::fold::FoldIdentityValue(graph.operation));

  for (u32 partition = 0u; partition < graph.partition_count; ++partition) {
    const u32 slot = graph.partition_fold_slots[partition];
    if (slot >= required_slots) {
      return fail("fold_graph_slot_out_of_range");
    }
    scratch_slots[slot] = partition_values[partition];
  }

  u32 final_slot = validation.final_slot;
  for (u32 edge_index = 0u; edge_index < graph.reduction_edge_count; ++edge_index) {
    const FoldGraphEdge& edge = graph.reduction_edges[edge_index];
    const u64 right = edge.right_is_padding ? edge.padding_value : scratch_slots[edge.right_slot];
    scratch_slots[edge.output_slot] =
        reduction::fold::CombineFoldValues(graph.operation,
                                           scratch_slots[edge.left_slot],
                                           right,
                                           graph.strict_float_reduction);
    final_slot = edge.output_slot;
  }

  const auto end = std::chrono::steady_clock::now();
  return FoldResult{
      .ok = true,
      .reason = "pass",
      .value = scratch_slots[final_slot],
      .slot_count = graph.partition_count,
      .operation = graph.operation,
      .fold_cost_ns = static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
      .fold_cost_measured = true,
  };
}

FoldResult FoldGraphReduce(const FoldGraphView graph,
                           const std::span<const u64> partition_values,
                           FoldSlots& scratch,
                           const AllocationPolicy allocation) {
  if (!EnsureFoldGraphScratch(scratch, graph, allocation)) {
    return FoldResult{
        .ok = false,
        .reason = "fold_graph_scratch_capacity_exceeded",
        .slot_count = graph.partition_count,
        .operation = graph.operation,
    };
  }
  return FoldGraphReduce(graph, partition_values, MutableFoldSlots(scratch));
}

} // namespace rund::kernel
