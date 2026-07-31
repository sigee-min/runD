#include "../../sort/block/metal.hpp"
#include "local.hpp"

#include <kernel/core/checked.hpp>
#include <limits>

namespace rund::node::accel::detail {

bool SortBlockShapeOk(const rund::kernel::SortPlan &plan,
                      rund::kernel::u32 &block_size,
                      rund::kernel::u32 &block_count,
                      rund::kernel::u64 &block_table_bytes) noexcept {
  if (plan.element_count == 0u ||
      plan.element_count > static_cast<rund::kernel::u64>(
                               std::numeric_limits<rund::kernel::u32>::max())) {
    return false;
  }
  block_size = kMetalSortBlockSize;
  const rund::kernel::u64 blocks =
      (plan.element_count + block_size - 1u) / block_size;
  if (blocks == 0u ||
      blocks > static_cast<rund::kernel::u64>(
                   std::numeric_limits<rund::kernel::u32>::max()) ||
      plan.bucket_count != kSortBucketCount || plan.radix_pass_count == 0u ||
      plan.radix_pass_count > 8u ||
      !rund::kernel::checked::mul(blocks, plan.bucket_count)) {
    return false;
  }
  const rund::kernel::u64 block_entries = blocks * plan.bucket_count;
  if (block_entries > static_cast<rund::kernel::u64>(
                          std::numeric_limits<rund::kernel::u32>::max()) ||
      !rund::kernel::checked::mul(block_entries, sizeof(rund::kernel::u32))) {
    return false;
  }
  block_count = static_cast<rund::kernel::u32>(blocks);
  block_table_bytes = block_entries * sizeof(rund::kernel::u32);
  return true;
}

} // namespace rund::node::accel::detail
