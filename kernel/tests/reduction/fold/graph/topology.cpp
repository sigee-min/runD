#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <string_view>

int RunFoldGraphTopologyContract() {
  rund::kernel::FoldGraph graph{};
  const rund::kernel::FoldGraphBuild missing =
      rund::kernel::BuildFoldGraph(graph,
                             5u,
                             rund::kernel::FoldOperation::FixedBinaryTreeHash,
                             rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} == "fold_graph_capacity_exceeded");
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 5u));
  const rund::kernel::FoldGraphBuild build =
      rund::kernel::BuildFoldGraph(graph,
                             5u,
                             rund::kernel::FoldOperation::FixedBinaryTreeHash,
                             rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(build.ok);
  TEST_ASSERT(build.no_allocation);
  const rund::kernel::FoldGraphView view = rund::kernel::ViewFoldGraph(graph);
  TEST_ASSERT(view.partition_count == 5u);
  TEST_ASSERT(view.partition_fold_slot_count == 5u);
  TEST_ASSERT(view.partition_fold_slots[0u] == 0u);
  TEST_ASSERT(view.partition_fold_slots[4u] == 4u);
  TEST_ASSERT(view.reduction_edge_count == 6u);
  TEST_ASSERT(view.node_count == 16u);
  TEST_ASSERT(view.program.node_count == 16u);
  TEST_ASSERT(view.program.edge_count == 6u);
  TEST_ASSERT(view.program.scratch_slot_count == rund::kernel::FoldGraphScratchSlotCount(view));
  TEST_ASSERT(view.dag_validated);
  TEST_ASSERT(view.slot_bounds_validated);
  TEST_ASSERT(view.padding_law_validated);
  TEST_ASSERT(view.fixed_binary_tree);
  TEST_ASSERT(view.primitive_standardized);
  TEST_ASSERT(!view.floating_point_allowed);
  TEST_ASSERT(!view.strict_floating_point);
  return 0;
}
