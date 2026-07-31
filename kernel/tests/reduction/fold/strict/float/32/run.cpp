#include "test/assert.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunStrictFloat32AssociativeContract();

namespace {

int RunStrictFloat32BasicContract() {
  const std::vector<rund::kernel::u64> float32_values{0x3F800000u, 0x40000000u};
  rund::kernel::FoldGraph graph{};
  TEST_ASSERT(rund::kernel::ReserveFoldGraph(graph, 4u));
  const rund::kernel::FoldGraphBuild default_float =
      rund::kernel::BuildFoldGraph(graph,
                             2u,
                             rund::kernel::FoldOperation::StrictFloat32Add,
                             rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(!default_float.ok);
  TEST_ASSERT(std::string_view{default_float.reason} == "floating_point_fold_forbidden");
  const rund::kernel::StrictFloatReductionPolicy fp32_policy = rund::kernel::StrictFloat32ReductionPolicy();
  const rund::kernel::FoldGraphBuild fp32_build =
      rund::kernel::BuildFoldGraph(graph,
                             2u,
                             rund::kernel::FoldOperation::StrictFloat32Add,
                             fp32_policy,
                             rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_build.ok);
  TEST_ASSERT(fp32_build.strict_floating_point);
  TEST_ASSERT(fp32_build.value_domain == rund::kernel::FoldValueDomain::Float32Strict);

  rund::kernel::FoldSlots scratch{};
  TEST_ASSERT(rund::kernel::ReserveFoldSlots(scratch, rund::kernel::FoldGraphScratchSlotCount(rund::kernel::ViewFoldGraph(graph))));
  const rund::kernel::FoldResult fp32_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(float32_values.data(),
                                                           float32_values.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_sum.ok);
  TEST_ASSERT(fp32_sum.value == 0x40400000u);

  const std::vector<rund::kernel::u64> fp32_nan{0x7FC00001u, 0x3F800000u};
  const rund::kernel::FoldResult fp32_nan_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp32_nan.data(), fp32_nan.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_nan_sum.ok);
  TEST_ASSERT(fp32_nan_sum.value == 0x7FC00000u);

  const std::vector<rund::kernel::u64> fp32_negative_zero{0x80000000u, 0x80000000u};
  const rund::kernel::FoldResult fp32_zero_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp32_negative_zero.data(),
                                                           fp32_negative_zero.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_zero_sum.ok);
  TEST_ASSERT(fp32_zero_sum.value == 0x00000000u);

  const std::vector<rund::kernel::u64> fp32_overflow{0x7F7FFFFFu, 0x7F7FFFFFu};
  const rund::kernel::FoldResult fp32_overflow_sum =
      rund::kernel::FoldGraphReduce(rund::kernel::ViewFoldGraph(graph),
                              std::span<const rund::kernel::u64>(fp32_overflow.data(),
                                                           fp32_overflow.size()),
                              scratch,
                              rund::kernel::AllocationPolicy::NoGrowth);
  TEST_ASSERT(fp32_overflow_sum.ok);
  TEST_ASSERT(fp32_overflow_sum.value == 0x7F800000u);
  return 0;
}

} // namespace

int RunStrictFloat32Contract() {
  if (const int rc = RunStrictFloat32BasicContract(); rc != 0) {
    return rc;
  }
  return RunStrictFloat32AssociativeContract();
}
