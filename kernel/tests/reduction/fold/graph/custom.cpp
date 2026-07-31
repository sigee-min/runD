#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <array>
#include <span>
#include <string_view>

int RunFoldGraphCustomContract() {
  const std::array<rund::kernel::u32, 4u> custom_slots{0u, 1u, 2u, 3u};
  const std::array<rund::kernel::FoldGraphEdge, 3u> custom_edges{
      rund::kernel::FoldGraphEdge{
          .level = 0u,
          .left_slot = 0u,
          .right_slot = 1u,
          .output_slot = 4u,
          .operation = rund::kernel::FoldOperation::SaturatingAdd,
      },
      rund::kernel::FoldGraphEdge{
          .level = 1u,
          .left_slot = 4u,
          .right_slot = 2u,
          .output_slot = 5u,
          .operation = rund::kernel::FoldOperation::SaturatingAdd,
      },
      rund::kernel::FoldGraphEdge{
          .level = 2u,
          .left_slot = 5u,
          .right_slot = 3u,
          .output_slot = 6u,
          .operation = rund::kernel::FoldOperation::SaturatingAdd,
      },
  };
  const rund::kernel::FoldGraphView custom_graph{
      .operation = rund::kernel::FoldOperation::SaturatingAdd,
      .value_domain = rund::kernel::FoldValueDomain::UnsignedInteger,
      .partition_count = 4u,
      .worker_local_slot_count = 4u,
      .global_ordered_slot_count = 4u,
      .scratch_slot_count = 7u,
      .final_slot = 6u,
      .fixed_binary_tree = false,
      .fixed_topological_order = true,
      .primitive_standardized = true,
      .partition_fold_slots = custom_slots.data(),
      .partition_fold_slot_count = static_cast<rund::kernel::u32>(custom_slots.size()),
      .reduction_edges = custom_edges.data(),
      .reduction_edge_count = static_cast<rund::kernel::u32>(custom_edges.size()),
  };
  const rund::kernel::FoldGraphValidationResult custom_validation =
      rund::kernel::ValidateFoldGraph(custom_graph);
  TEST_ASSERT(custom_validation.ok);
  TEST_ASSERT(custom_validation.final_slot == 6u);
  rund::kernel::FoldSlots custom_scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(custom_scratch, rund::kernel::FoldGraphScratchSlotCount(custom_graph)));
  const std::array<rund::kernel::u64, 4u> custom_values{10u, 20u, 30u, 40u};
  const rund::kernel::FoldResult custom_result =
      rund::kernel::FoldGraphReduce(custom_graph,
                              std::span<const rund::kernel::u64>(custom_values.data(), custom_values.size()),
                              custom_scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult custom_repeat =
      rund::kernel::FoldGraphReduce(custom_graph,
                              std::span<const rund::kernel::u64>(custom_values.data(), custom_values.size()),
                              custom_scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(custom_result.ok);
  TEST_ASSERT(custom_repeat.ok);
  TEST_ASSERT(custom_result.value == 100u);
  TEST_ASSERT(custom_repeat.value == custom_result.value);

  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 5u));
  const rund::kernel::FoldGraphBuild unsupported =
      rund::kernel::BuildFoldGraph(graph,
                             5u,
                             static_cast<rund::kernel::FoldOperation>(255u),
                             rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(!unsupported.ok);
  TEST_ASSERT(std::string_view{unsupported.reason} == "unsupported_fold_operation");
  return 0;
}
