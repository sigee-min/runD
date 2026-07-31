#include "local.hpp"

namespace program_compute_contract {

int HistogramReject() {
  rund::kernel::HistogramDesc unknown_index = U32Histogram();
  unknown_index.index = static_cast<rund::kernel::HistogramIndex>(0u);
  const rund::kernel::HistogramPlan unknown_index_plan =
      rund::kernel::PlanHistogram(unknown_index);
  TEST_ASSERT(!unknown_index_plan.ok);
  TEST_ASSERT(std::string_view{unknown_index_plan.reason} ==
              "compute_histogram_index_unsupported");

  rund::kernel::HistogramDesc unknown_count = U32Histogram();
  unknown_count.count = static_cast<rund::kernel::HistogramCount>(0u);
  const rund::kernel::HistogramPlan unknown_count_plan =
      rund::kernel::PlanHistogram(unknown_count);
  TEST_ASSERT(!unknown_count_plan.ok);
  TEST_ASSERT(std::string_view{unknown_count_plan.reason} ==
              "compute_histogram_count_unsupported");

  rund::kernel::HistogramDesc zero_count = U32Histogram();
  zero_count.element_count = 0u;
  const rund::kernel::HistogramPlan zero_count_plan =
      rund::kernel::PlanHistogram(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_histogram_count_zero");

  rund::kernel::HistogramDesc zero_bins = U32Histogram();
  zero_bins.bin_count = 0u;
  const rund::kernel::HistogramPlan zero_bins_plan =
      rund::kernel::PlanHistogram(zero_bins);
  TEST_ASSERT(!zero_bins_plan.ok);
  TEST_ASSERT(std::string_view{zero_bins_plan.reason} ==
              "compute_histogram_bin_count_zero");

  rund::kernel::HistogramDesc overflow = U32Histogram();
  overflow.element_count =
      static_cast<rund::kernel::u64>(
          std::numeric_limits<rund::kernel::u32>::max()) +
      1u;
  const rund::kernel::HistogramPlan overflow_plan =
      rund::kernel::PlanHistogram(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_histogram_count_overflow");
  return 0;
}

}  // namespace program_compute_contract
