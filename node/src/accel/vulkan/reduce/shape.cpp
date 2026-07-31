#include "local.hpp"

#include <kernel/program/compute/reduce/plan.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool VulkanReduceIndexRangeOk(
    const rund::kernel::ReducePlan& plan) noexcept {
  constexpr rund::kernel::u64 max_index =
      std::numeric_limits<std::uint32_t>::max();
  if (plan.element_count > max_index ||
      plan.first_pass_group_count > max_index) {
    return false;
  }
  if (rund::kernel::ReduceUsesWidePartials(plan.op)) {
    return plan.pass_count == (plan.first_pass_group_count > 1u ? 2u : 1u);
  }
  rund::kernel::u64 current = plan.element_count;
  rund::kernel::u64 write_offset = 0u;
  for (rund::kernel::u64 pass = 0u; pass < plan.pass_count; ++pass) {
    const rund::kernel::u64 next =
        rund::kernel::ReduceGroupCount(current, plan.block_size);
    const bool final_pass = next == 1u;
    if (next == 0u || next > max_index) { return false; }
    if (!final_pass) {
      if (write_offset > max_index || next > max_index - write_offset) {
        return false;
      }
      write_offset += next;
    }
    current = next;
  }
  return true;
}
#endif

}  // namespace rund::node::accel::detail
