#include "local.hpp"

namespace program_compute_contract {

int ScanReject() {
  rund::kernel::ScanDesc zero_count = U32Scan();
  zero_count.element_count = 0u;
  const rund::kernel::ScanPlan zero_count_plan =
      rund::kernel::PlanScan(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_scan_count_zero");

  rund::kernel::ScanDesc unknown_op = U32Scan();
  unknown_op.op = static_cast<rund::kernel::ScanOp>(0u);
  const rund::kernel::ScanPlan unknown_op_plan =
      rund::kernel::PlanScan(unknown_op);
  TEST_ASSERT(!unknown_op_plan.ok);
  TEST_ASSERT(std::string_view{unknown_op_plan.reason} ==
              "compute_scan_op_unsupported");

  rund::kernel::ScanDesc unknown_count_source = U32Scan();
  unknown_count_source.count_source =
      static_cast<rund::kernel::ComputeCountSource>(255u);
  const rund::kernel::ScanPlan unknown_count_source_plan =
      rund::kernel::PlanScan(unknown_count_source);
  TEST_ASSERT(!unknown_count_source_plan.ok);
  TEST_ASSERT(std::string_view{unknown_count_source_plan.reason} ==
              "compute_scan_count_source_unsupported");

  rund::kernel::ScanDesc unknown_element = U32Scan();
  unknown_element.element = static_cast<rund::kernel::ScanElement>(0u);
  const rund::kernel::ScanPlan unknown_element_plan =
      rund::kernel::PlanScan(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_scan_element_unsupported");

  rund::kernel::ScanDesc zero_block = U32Scan();
  zero_block.block_size = 0u;
  const rund::kernel::ScanPlan zero_block_plan =
      rund::kernel::PlanScan(zero_block);
  TEST_ASSERT(!zero_block_plan.ok);
  TEST_ASSERT(std::string_view{zero_block_plan.reason} ==
              "compute_scan_block_invalid");

  rund::kernel::ScanDesc temp_overflow = U32Scan();
  temp_overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 4u) + 1u;
  const rund::kernel::ScanPlan overflow_plan =
      rund::kernel::PlanScan(temp_overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_scan_temp_overflow");
  return 0;
}

} // namespace program_compute_contract
