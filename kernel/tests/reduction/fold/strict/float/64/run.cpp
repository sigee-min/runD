#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <span>
#include <vector>

int RunStrictFloat64AssociativeContract();

namespace {

int RunStrictFloat64BasicContract() {
  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 4u));
  TEST_ASSERT(rund::kernel::BuildFoldGraph(graph,
                                     2u,
                                     rund::kernel::FoldOperation::StrictFloat64Add,
                                     rund::kernel::StrictFloat64ReductionPolicy(),
                                     rund::kernel::AllocationPolicy::NoGrowth)
                  .ok);
  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch, rund::kernel::FoldGraphScratchSlotCount(rund::kernel::ViewFoldGraph(graph))));
  const std::vector<rund::kernel::u64> values{0x3FF0000000000000ull, 0x4000000000000000ull};
  const rund::kernel::FoldResult sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(values.data(), values.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(sum.ok);
  TEST_ASSERT(sum.value == 0x4008000000000000ull);

  const std::vector<rund::kernel::u64> nan{0x7FF8000000000001ull, 0x3FF0000000000000ull};
  const rund::kernel::FoldResult nan_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(nan.data(), nan.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(nan_sum.ok);
  TEST_ASSERT(nan_sum.value == 0x7FF8000000000000ull);

  const std::vector<rund::kernel::u64> negative_zero{0x8000000000000000ull, 0x8000000000000000ull};
  const rund::kernel::FoldResult zero_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(negative_zero.data(), negative_zero.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(zero_sum.ok);
  TEST_ASSERT(zero_sum.value == 0x0000000000000000ull);

  const std::vector<rund::kernel::u64> overflow{0x7FEFFFFFFFFFFFFFull, 0x7FEFFFFFFFFFFFFFull};
  const rund::kernel::FoldResult overflow_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(overflow.data(), overflow.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(overflow_sum.ok);
  TEST_ASSERT(overflow_sum.value == 0x7FF0000000000000ull);
  return 0;
}

} // namespace

int RunStrictFloat64Contract() {
  if (const int rc = RunStrictFloat64BasicContract(); rc != 0) {
    return rc;
  }
  return RunStrictFloat64AssociativeContract();
}
