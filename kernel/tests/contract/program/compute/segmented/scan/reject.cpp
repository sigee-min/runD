#include "local.hpp"

namespace program_compute_contract {

int SegmentedScanReject() {
  rund::kernel::SegmentedScanDesc zero_count = U32SegmentedScan();
  zero_count.element_count = 0u;
  const rund::kernel::SegmentedScanPlan zero_count_plan =
      rund::kernel::PlanSegmentedScan(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_segmented_scan_count_zero");

  rund::kernel::SegmentedScanDesc unknown_op = U32SegmentedScan();
  unknown_op.op = static_cast<rund::kernel::SegmentedScanOp>(0u);
  const rund::kernel::SegmentedScanPlan unknown_op_plan =
      rund::kernel::PlanSegmentedScan(unknown_op);
  TEST_ASSERT(!unknown_op_plan.ok);
  TEST_ASSERT(std::string_view{unknown_op_plan.reason} ==
              "compute_segmented_scan_op_unsupported");

  rund::kernel::SegmentedScanDesc unknown_element = U32SegmentedScan();
  unknown_element.element =
      static_cast<rund::kernel::SegmentedScanElement>(0u);
  const rund::kernel::SegmentedScanPlan unknown_element_plan =
      rund::kernel::PlanSegmentedScan(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_segmented_scan_element_unsupported");

  rund::kernel::SegmentedScanDesc zero_block = U32SegmentedScan();
  zero_block.block_size = 0u;
  const rund::kernel::SegmentedScanPlan zero_block_plan =
      rund::kernel::PlanSegmentedScan(zero_block);
  TEST_ASSERT(!zero_block_plan.ok);
  TEST_ASSERT(std::string_view{zero_block_plan.reason} ==
              "compute_segmented_scan_block_invalid");

  rund::kernel::SegmentedScanDesc overflow = U32SegmentedScan();
  overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 8u) + 1u;
  const rund::kernel::SegmentedScanPlan overflow_plan =
      rund::kernel::PlanSegmentedScan(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_segmented_scan_temp_overflow");
  return 0;
}

} // namespace program_compute_contract
