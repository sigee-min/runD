#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <span>
#include <vector>

int RunStrictFloat32AssociativeContract() {
  const rund::kernel::StrictFloatReductionPolicy fp32_policy = rund::kernel::StrictFloat32ReductionPolicy();
  const std::vector<rund::kernel::u64> fp32_associative{0x3F800000u, 0x33000000u, 0xBF800000u};
  rund::kernel::FoldGraph graph{};
  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 4u));
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     3u,
                                     rund::kernel::FoldOperation::StrictFloat32Add,
                                     fp32_policy,
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch, rund::kernel::FoldGraphScratchSlotCount(rund::kernel::ViewFoldGraph(graph))));
  const rund::kernel::FoldResult fp32_assoc_first =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp32_associative.data(),
                                                           fp32_associative.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult fp32_assoc_second =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp32_associative.data(),
                                                           fp32_associative.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_assoc_first.ok);
  TEST_ASSERT(fp32_assoc_second.ok);
  TEST_ASSERT(fp32_assoc_first.value == fp32_assoc_second.value);
  return 0;
}
