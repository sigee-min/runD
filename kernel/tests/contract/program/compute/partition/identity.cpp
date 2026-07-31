#include "test/assert.hpp"

#include "local.hpp"

namespace program_compute_contract {
namespace {

int test_compute_partition_descriptor_hash_is_deterministic_and_field_sensitive() {
  const rund::kernel::PartitionDesc desc =
      partition_contract::U32Partition();
  const rund::kernel::PartitionHash first =
      rund::kernel::HashPartition(desc);
  const rund::kernel::PartitionHash second =
      rund::kernel::HashPartition(desc);
  rund::kernel::PartitionDesc changed = desc;
  changed.element_count += 1u;
  const rund::kernel::PartitionHash changed_hash =
      rund::kernel::HashPartition(changed);

  TEST_ASSERT(first.hi == second.hi);
  TEST_ASSERT(first.lo == second.lo);
  TEST_ASSERT(first.hi != 0u || first.lo != 0u);
  TEST_ASSERT(first.hi != changed_hash.hi || first.lo != changed_hash.lo);
  return 0;
}

}  // namespace

int RunPartitionIdentityContract() {
  return test_compute_partition_descriptor_hash_is_deterministic_and_field_sensitive();
}

}  // namespace program_compute_contract
