#include "../projection/internal.hpp"

namespace rund::compute::detail::graph_reduce {

residency::FrameRegion resource_region(const residency::Pool &pool,
                                       const residency::TiledGraphPlan &plan,
                                       const std::uint32_t resource,
                                       const std::uint32_t bank) noexcept {
  const residency::TiledGraphResource *const declared = plan.resource(resource);
  const residency::PoolPhysicalOwner *const owner =
      declared == nullptr ? nullptr : pool.graph_owner(declared->physical_id);
  return owner == nullptr || bank >= residency::Pool::BankCount
             ? residency::FrameRegion{}
             : owner->bank_regions[bank];
}

bool resource_cache_regions(
    const residency::Pool &pool, const residency::TiledGraphPlan &plan,
    const std::uint32_t resource,
    std::array<residency::FrameRegion, residency::Pool::BankCount> &regions,
    std::size_t &count) noexcept {
  regions = {};
  count = 0u;
  const residency::TiledGraphResource *const declared = plan.resource(resource);
  const residency::PoolPhysicalOwner *const owner =
      declared == nullptr ? nullptr : pool.graph_owner(declared->physical_id);
  if (owner == nullptr || owner->arena == nullptr) {
    return false;
  }
  regions = owner->cache_regions;
  count = regions.size();
  return true;
}

bool copy_keys(const std::span<const residency::PageUse> uses,
               const residency::GraphMaterialization materialization,
               const std::span<residency::CacheKey> keys) noexcept {
  if (uses.size() != keys.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < uses.size(); ++index) {
    if (!residency::project_graph_cache_key(materialization, uses[index].key,
                                            keys[index])) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce
