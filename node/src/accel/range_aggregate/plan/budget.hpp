#pragma once

#include "../model/capability.hpp"

#include <algorithm>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr rund::kernel::u32
SharedRadiusCapacity(const RangeCaps &capabilities,
                     const rund::kernel::u32 width,
                     const rund::kernel::u32 element_bytes) noexcept {
  const rund::kernel::u32 occupancy =
      capabilities.shared_memory_occupancy_budget();
  if (occupancy == 0u || element_bytes == 0u) {
    return 0u;
  }
  const rund::kernel::u64 element_capacity =
      capabilities.shared_memory_limit() / occupancy / element_bytes;
  if (element_capacity <= width) {
    return 0u;
  }
  const rund::kernel::u64 radius_capacity = (element_capacity - width) / 2u;
  return static_cast<rund::kernel::u32>(
      std::min<rund::kernel::u64>(width, radius_capacity));
}

[[nodiscard]] constexpr bool
SharedBudgetFits(const RangeCaps &capabilities,
                 const rund::kernel::u64 shared_bytes) noexcept {
  const rund::kernel::u32 occupancy =
      capabilities.shared_memory_occupancy_budget();
  return occupancy != 0u &&
         shared_bytes <= capabilities.shared_memory_limit() / occupancy;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
