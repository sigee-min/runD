#include "local.hpp"

namespace program_compute_contract {

int SegmentedScanIdentity() {
  const rund::kernel::SegmentedScanPlan first =
      rund::kernel::PlanSegmentedScan(U32SegmentedScan());
  const rund::kernel::SegmentedScanPlan second =
      rund::kernel::PlanSegmentedScan(U32SegmentedScan());
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.element_count == second.element_count);
  TEST_ASSERT(first.element_bytes == second.element_bytes);
  TEST_ASSERT(first.block_size == second.block_size);
  TEST_ASSERT(first.block_count == second.block_count);
  TEST_ASSERT(first.pass_count == second.pass_count);
  TEST_ASSERT(first.temp_value_bytes == second.temp_value_bytes);
  TEST_ASSERT(first.temp_head_bytes == second.temp_head_bytes);
  TEST_ASSERT(first.temp_bytes == second.temp_bytes);
  TEST_ASSERT(std::string_view{first.reason} ==
              std::string_view{second.reason});

  const rund::kernel::SegmentedScanDesc desc = U32SegmentedScan();
  const rund::kernel::SegmentedScanHash first_hash =
      rund::kernel::HashSegmentedScan(desc);
  const rund::kernel::SegmentedScanHash second_hash =
      rund::kernel::HashSegmentedScan(desc);
  rund::kernel::SegmentedScanDesc changed = desc;
  changed.block_size = 8u;
  const rund::kernel::SegmentedScanHash changed_hash =
      rund::kernel::HashSegmentedScan(changed);
  rund::kernel::SegmentedScanDesc inclusive = desc;
  inclusive.op = rund::kernel::SegmentedScanOp::InclusiveSum;
  const rund::kernel::SegmentedScanHash inclusive_hash =
      rund::kernel::HashSegmentedScan(inclusive);

  TEST_ASSERT(first_hash.hi == second_hash.hi);
  TEST_ASSERT(first_hash.lo == second_hash.lo);
  TEST_ASSERT(first_hash.hi != 0u || first_hash.lo != 0u);
  TEST_ASSERT(first_hash.hi != changed_hash.hi ||
              first_hash.lo != changed_hash.lo);
  TEST_ASSERT(first_hash.hi != inclusive_hash.hi ||
              first_hash.lo != inclusive_hash.lo);
  return 0;
}

} // namespace program_compute_contract
