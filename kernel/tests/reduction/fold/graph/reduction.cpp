#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <array>
#include <span>
#include <vector>

int RunFoldGraphReductionContract() {
  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 8u));
  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch,
                                             rund::kernel::FoldGraphFixedBinaryTreeScratchSlotCount(8u)));
  const std::array<rund::kernel::FoldOperation, 5u> graph_operations{
      rund::kernel::FoldOperation::Xor,
      rund::kernel::FoldOperation::Max,
      rund::kernel::FoldOperation::Min,
      rund::kernel::FoldOperation::SaturatingAdd,
      rund::kernel::FoldOperation::FixedBinaryTreeHash,
  };
  const std::array<rund::kernel::u32, 2u> partition_counts{5u, 8u};
  for (const rund::kernel::u32 partition_count : partition_counts) {
    std::vector<rund::kernel::u64> values(partition_count, 0u);
    for (rund::kernel::u32 index = 0u; index < partition_count; ++index) {
      values[index] = static_cast<rund::kernel::u64>((index + 1u) * 0x10u);
    }
    for (const rund::kernel::FoldOperation operation : graph_operations) {
      TEST_ASSERT(rund::kernel::BuildFoldGraph(graph, partition_count, operation, rund::kernel::AllocationPolicy::NoGrowth).ok);
      const rund::kernel::FoldResult graph_result =
          rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                                  std::span<const rund::kernel::u64>(values.data(), values.size()),
                                  scratch,
                                  rund::kernel::AllocationPolicy::NoGrowth);
      const rund::kernel::FoldResult ordered_result =
          rund::kernel::FoldOrderedSlots(std::span<const rund::kernel::u64>(values.data(), values.size()), operation);
      TEST_ASSERT(graph_result.ok);
      TEST_ASSERT(graph_result.fold_cost_measured);
      TEST_ASSERT(ordered_result.ok);
      TEST_ASSERT(graph_result.value == ordered_result.value);
    }
  }

  const std::vector<rund::kernel::u64> float32_values{0x3F800000u, 0x40000000u, 0x40400000u};
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     static_cast<rund::kernel::u32>(float32_values.size()),
                                     rund::kernel::FoldOperation::StrictFloat32Add,
                                     rund::kernel::StrictFloat32ReductionPolicy(),
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  const rund::kernel::FoldResult float32_graph_result =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(float32_values.data(), float32_values.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult float32_ordered_result =
      rund::kernel::FoldStrictOrderedSlots(std::span<const rund::kernel::u64>(float32_values.data(),
                                                                        float32_values.size()),
                                     rund::kernel::FoldOperation::StrictFloat32Add,
                                     rund::kernel::StrictFloat32ReductionPolicy());
  TEST_ASSERT(float32_graph_result.ok);
  TEST_ASSERT(float32_ordered_result.ok);
  TEST_ASSERT(float32_graph_result.value == float32_ordered_result.value);

  const std::vector<rund::kernel::u64> float64_values{
      0x3FF0000000000000ull,
      0x4000000000000000ull,
      0x4008000000000000ull,
      0x4010000000000000ull,
  };
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     static_cast<rund::kernel::u32>(float64_values.size()),
                                     rund::kernel::FoldOperation::StrictFloat64Add,
                                     rund::kernel::StrictFloat64ReductionPolicy(),
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  const rund::kernel::FoldResult float64_graph_result =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(float64_values.data(), float64_values.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult float64_ordered_result =
      rund::kernel::FoldStrictOrderedSlots(std::span<const rund::kernel::u64>(float64_values.data(),
                                                                        float64_values.size()),
                                     rund::kernel::FoldOperation::StrictFloat64Add,
                                     rund::kernel::StrictFloat64ReductionPolicy());
  TEST_ASSERT(float64_graph_result.ok);
  TEST_ASSERT(float64_ordered_result.ok);
  TEST_ASSERT(float64_graph_result.value == float64_ordered_result.value);
  return 0;
}
