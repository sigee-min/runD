#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <span>
#include <vector>

int RunStrictFloat64AssociativeContract() {
  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 6u));
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     3u,
                                     rund::kernel::FoldOperation::StrictFloat64Add,
                                     rund::kernel::StrictFloat64ReductionPolicy(),
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch, rund::kernel::FoldGraphScratchSlotCount(rund::kernel::ViewFoldGraph(graph))));
  const std::vector<rund::kernel::u64> fp64_associative_zero{
      0x3FF0000000000000ull,
      0x3CA0000000000000ull,
      0xBFF0000000000000ull,
  };
  const rund::kernel::FoldResult zero_first =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp64_associative_zero.data(),
                                                           fp64_associative_zero.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult zero_second =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp64_associative_zero.data(),
                                                           fp64_associative_zero.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(zero_first.ok);
  TEST_ASSERT(zero_second.ok);
  TEST_ASSERT(zero_first.value == zero_second.value);
  TEST_ASSERT(zero_first.value == 0x0000000000000000ull);

  const std::vector<rund::kernel::u64> fp64_associative_small{
      0x3FF0000000000000ull,
      0xBFF0000000000000ull,
      0x3CA0000000000000ull,
  };
  const rund::kernel::FoldResult small_first =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp64_associative_small.data(),
                                                           fp64_associative_small.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  const rund::kernel::FoldResult small_second =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp64_associative_small.data(),
                                                           fp64_associative_small.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(small_first.ok);
  TEST_ASSERT(small_second.ok);
  TEST_ASSERT(small_first.value == small_second.value);
  TEST_ASSERT(small_first.value == 0x3CA0000000000000ull);
  return 0;
}
