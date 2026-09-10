#include "internal.hpp"

namespace rund::compute::detail::residency {

FrameRole frame_role(const GraphResourceRole role) noexcept {
  return role == GraphResourceRole::Input          ? FrameRole::Input
         : role == GraphResourceRole::Intermediate ? FrameRole::Intermediate
                                                   : FrameRole::Output;
}

bool graph_classes_match(
    const Pool &pool,
    const std::span<const TiledGraphPhysicalClass> classes) noexcept {
  if (pool.graph_owners.size() != classes.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const TiledGraphPhysicalClass &declared = classes[index];
    const PoolPhysicalOwner &owner = pool.graph_owners[index];
    if (owner.physical_id != declared.physical_id ||
        owner.color != declared.color ||
        owner.physical_class.type != declared.type ||
        owner.physical_class.format != declared.format ||
        owner.physical_class.page_bytes != declared.page_bytes ||
        owner.physical_class.role != frame_role(declared.role)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
