#include "../../sort/block/vulkan.hpp"
#include "local/api.hpp"

#include <kernel/core/checked.hpp>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool SortBlockShapeOk(const rund::kernel::SortPlan &plan,
                      rund::kernel::u32 &block_count,
                      rund::kernel::u64 &block_table_bytes) noexcept {
  if (plan.element_count == 0u ||
      plan.element_count > static_cast<rund::kernel::u64>(
                               std::numeric_limits<rund::kernel::u32>::max())) {
    return false;
  }
  const rund::kernel::u64 blocks =
      (plan.element_count + kVulkanSortBlockSize - 1u) / kVulkanSortBlockSize;
  if (blocks == 0u ||
      blocks > static_cast<rund::kernel::u64>(
                   std::numeric_limits<rund::kernel::u32>::max()) ||
      plan.bucket_count != kSortBucketCount ||
      !rund::kernel::checked::mul(blocks, plan.bucket_count)) {
    return false;
  }
  const rund::kernel::u64 block_entries = blocks * plan.bucket_count;
  const rund::kernel::u64 max_entries = static_cast<rund::kernel::u64>(
      std::numeric_limits<rund::kernel::u32>::max());
  if (block_entries > max_entries - plan.bucket_count ||
      !rund::kernel::checked::mul(block_entries, sizeof(rund::kernel::u32))) {
    return false;
  }
  block_count = static_cast<rund::kernel::u32>(blocks);
  block_table_bytes = block_entries * sizeof(rund::kernel::u32);
  return true;
}
#endif

} // namespace rund::node::accel::detail
