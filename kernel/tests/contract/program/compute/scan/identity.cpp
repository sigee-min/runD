#include "local.hpp"

namespace program_compute_contract {

int ScanIdentity() {
  const rund::kernel::ScanPlan first =
      rund::kernel::PlanScan(U32Scan());
  const rund::kernel::ScanPlan second =
      rund::kernel::PlanScan(U32Scan());

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.element_count == second.element_count);
  TEST_ASSERT(first.element_bytes == second.element_bytes);
  TEST_ASSERT(first.block_size == second.block_size);
  TEST_ASSERT(first.block_count == second.block_count);
  TEST_ASSERT(first.pass_count == second.pass_count);
  TEST_ASSERT(first.temp_bytes == second.temp_bytes);
  TEST_ASSERT(std::string_view{first.reason} ==
              std::string_view{second.reason});

  const rund::kernel::ScanDesc desc = U32Scan();
  const rund::kernel::ScanHash first_hash =
      rund::kernel::HashScan(desc);
  const rund::kernel::ScanHash second_hash =
      rund::kernel::HashScan(desc);
  rund::kernel::ScanDesc changed = desc;
  changed.block_size = 8u;
  const rund::kernel::ScanHash changed_hash =
      rund::kernel::HashScan(changed);
  rund::kernel::ScanDesc inclusive = desc;
  inclusive.op = rund::kernel::ScanOp::InclusiveSum;
  const rund::kernel::ScanHash inclusive_hash =
      rund::kernel::HashScan(inclusive);
  rund::kernel::ScanDesc bounded = desc;
  bounded.count_source = rund::kernel::ComputeCountSource::BufferU32;
  const rund::kernel::ScanHash bounded_hash =
      rund::kernel::HashScan(bounded);

  TEST_ASSERT(first_hash.hi == second_hash.hi);
  TEST_ASSERT(first_hash.lo == second_hash.lo);
  TEST_ASSERT(first_hash.hi != 0u || first_hash.lo != 0u);
  TEST_ASSERT(first_hash.hi != changed_hash.hi ||
              first_hash.lo != changed_hash.lo);
  TEST_ASSERT(first_hash.hi != inclusive_hash.hi ||
              first_hash.lo != inclusive_hash.lo);
  TEST_ASSERT(first_hash.hi != bounded_hash.hi ||
              first_hash.lo != bounded_hash.lo);
  return 0;
}

} // namespace program_compute_contract
