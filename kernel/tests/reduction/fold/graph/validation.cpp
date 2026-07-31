#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunFoldGraphValidationContract() {
  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 5u));
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  const rund::kernel::FoldGraphView view = rund::kernel::ViewFoldGraph(graph);
  rund::kernel::FoldGraphView malformed_final_slot = view;
  malformed_final_slot.final_slot = 0u;
  const rund::kernel::FoldGraphValidationResult malformed_final_validation =
      rund::kernel::ValidateFoldGraph(malformed_final_slot);
  TEST_ASSERT(!malformed_final_validation.ok);
  TEST_ASSERT(std::string_view{malformed_final_validation.reason} ==
              "fold_graph_final_slot_mismatch");

  rund::kernel::FoldGraphView forged_validated_view = view;
  forged_validated_view.final_slot = 0u;
  forged_validated_view.dag_validated = true;
  forged_validated_view.slot_bounds_validated = true;
  forged_validated_view.padding_law_validated = true;
  forged_validated_view.program.dag_validated = true;
  forged_validated_view.program.slot_bounds_validated = true;
  forged_validated_view.program.padding_law_validated = true;
  const rund::kernel::FoldGraphValidationResult forged_validated_validation =
      rund::kernel::ValidateFoldGraph(forged_validated_view);
  TEST_ASSERT(!forged_validated_validation.ok);
  TEST_ASSERT(std::string_view{forged_validated_validation.reason} ==
              "fold_graph_final_slot_mismatch");

  graph.reduction_edges[0u].padding_law = rund::kernel::FoldPaddingLaw::Identity;
  const rund::kernel::FoldGraphValidationResult malformed_padding_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!malformed_padding_validation.ok);
  TEST_ASSERT(std::string_view{malformed_padding_validation.reason} ==
              "fold_graph_padding_law_mismatch");

  const std::vector<rund::kernel::u64> values{0x10u, 0x20u, 0x30u, 0x40u, 0x50u};
  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch, rund::kernel::FoldGraphScratchSlotCount(view)));
  const std::size_t scratch_capacity = scratch.values.capacity();
  const rund::kernel::FoldResult malformed_no_growth_reduce =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(values.data(), values.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(!malformed_no_growth_reduce.ok);
  TEST_ASSERT(std::string_view{malformed_no_growth_reduce.reason} ==
              "fold_graph_padding_law_mismatch");
  TEST_ASSERT(scratch.values.capacity() == scratch_capacity);

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.partition_fold_slots[1u] = graph.partition_fold_slots[0u];
  const rund::kernel::FoldGraphValidationResult duplicate_partition_slot_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!duplicate_partition_slot_validation.ok);
  TEST_ASSERT(std::string_view{duplicate_partition_slot_validation.reason} ==
              "fold_graph_slot_out_of_range");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.reduction_edges[0u].left_slot = graph.scratch_slot_count;
  const rund::kernel::FoldGraphValidationResult unavailable_input_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!unavailable_input_validation.ok);
  TEST_ASSERT(std::string_view{unavailable_input_validation.reason} ==
              "fold_graph_edge_input_unavailable");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.reduction_edges[1u].output_slot = graph.reduction_edges[0u].output_slot;
  const rund::kernel::FoldGraphValidationResult redefined_output_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!redefined_output_validation.ok);
  TEST_ASSERT(std::string_view{redefined_output_validation.reason} ==
              "fold_graph_edge_output_redefined");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.nodes[0u].kind = rund::kernel::FoldGraphNodeKind::Reduction;
  const rund::kernel::FoldGraphValidationResult malformed_node_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!malformed_node_validation.ok);
  TEST_ASSERT(std::string_view{malformed_node_validation.reason} ==
              "fold_graph_node_mismatch");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.nodes[0u].slot = graph.reduction_edges[0u].output_slot;
  const rund::kernel::FoldGraphValidationResult partition_node_slot_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!partition_node_slot_validation.ok);
  TEST_ASSERT(std::string_view{partition_node_slot_validation.reason} ==
              "fold_graph_node_mismatch");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.nodes[graph.partition_count * 2u].left_slot = graph.scratch_slot_count - 1u;
  const rund::kernel::FoldGraphValidationResult reduction_node_edge_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!reduction_node_edge_validation.ok);
  TEST_ASSERT(std::string_view{reduction_node_edge_validation.reason} ==
              "fold_graph_node_mismatch");

  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     5u,
                                     rund::kernel::FoldOperation::FixedBinaryTreeHash,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  graph.nodes[0u].kind = static_cast<rund::kernel::FoldGraphNodeKind>(255u);
  const rund::kernel::FoldGraphValidationResult malformed_node_kind_validation =
      rund::kernel::ValidateFoldGraph(rund::kernel::ViewFoldGraph(graph));
  TEST_ASSERT(!malformed_node_kind_validation.ok);
  TEST_ASSERT(std::string_view{malformed_node_kind_validation.reason} ==
              "fold_graph_node_mismatch");

  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 3u));
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     3u,
                                     rund::kernel::FoldOperation::StrictFloat32Add,
                                     rund::kernel::StrictFloat32ReductionPolicy(),
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  rund::kernel::FoldGraphView missing_floating_point_law = rund::kernel::ViewFoldGraph(graph);
  missing_floating_point_law.strict_float_reduction.valid = false;
  const rund::kernel::FoldGraphValidationResult missing_floating_point_law_validation =
      rund::kernel::ValidateFoldGraph(missing_floating_point_law);
  TEST_ASSERT(!missing_floating_point_law_validation.ok);
  TEST_ASSERT(std::string_view{missing_floating_point_law_validation.reason} ==
              "floating_point_law_required");
  return 0;
}
