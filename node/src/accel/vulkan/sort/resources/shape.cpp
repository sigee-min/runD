#include <accel/check.hpp>

#include "../local/api.hpp"
#include "../../../sort/block/bucket.hpp"

#include "../../collective/chunk.hpp"
#include "../../../sort/shape.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck ValidateVulkanSortPrepareShape(
    VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
    const rund::kernel::SortPlan &plan, const SortBinds &bindings,
    VulkanSortPrepareShape &shape) {
  if (!SortShapeOk(desc, plan, bindings) ||
      !SortBlockShapeOk(plan, shape.block_count, shape.block_table_bytes) ||
      plan.radix_pass_count == 0u || plan.radix_pass_count > kMaxSortPasses ||
      plan.element_count > static_cast<rund::kernel::u64>(
                               std::numeric_limits<std::uint32_t>::max()) ||
      plan.bucket_count > static_cast<rund::kernel::u64>(
                              std::numeric_limits<std::uint32_t>::max()) ||
      adapter.max_dispatch_groups == 0u) {
    SetVulkanLastError(adapter, "compute_sort_invalid");
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }

  shape.count_bytes =
      shape.block_table_bytes + kSortBucketCount * sizeof(rund::kernel::u32);
  shape.chunk_count = static_cast<rund::kernel::u32>(
      CeilGroups(shape.block_count, adapter.max_dispatch_groups));
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
