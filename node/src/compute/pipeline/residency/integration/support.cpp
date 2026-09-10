#include "local.hpp"

#include "../../../device/residency/pool.hpp"

namespace rund::compute::detail {

[[nodiscard]] bool graph_frame_region(const residency::Pool &pool,
                                      const std::uint32_t bank,
                                      const std::uint32_t physical_id,
                                      residency::FrameRegion &region) noexcept {
  region = {};
  const residency::PoolPhysicalOwner *const owner =
      pool.graph_owner(physical_id);
  if (owner == nullptr || bank >= residency::Pool::BankCount) {
    return false;
  }
  region = owner->bank_regions[bank];
  return region.count == pool.layout.frame_capacity;
}

} // namespace rund::compute::detail
