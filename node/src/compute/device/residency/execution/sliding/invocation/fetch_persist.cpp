#include "../internal.hpp"

namespace rund::compute::detail::residency::execution {

bool SlidingInvocation::use_bytes(
    const SlidingProjection &projection,
    const std::span<const residency::PageUse> uses, const std::size_t use,
    std::uint64_t &bytes) const noexcept {
  bytes = 0u;
  if (!*this || use >= projection.use_count || use >= uses.size()) {
    return false;
  }
  if (writes(uses[use].access) && uses[use].dirty.bytes != 0u) {
    bytes = uses[use].dirty.bytes;
    return true;
  }
  return topology_ == SlidingTopology::Direct
             ? direct_->input_bytes(uses[use].key.page, bytes)
             : graph_.page_bytes(uses[use].key.resource, uses[use].key.page,
                                 bytes);
}

bool SlidingInvocation::fetch_use(
    const SlidingProjection &projection,
    const std::span<const residency::PageUse> uses,
    const std::size_t use) const noexcept {
  if (!*this || use >= projection.use_count || use >= uses.size() ||
      !reads(uses[use].access)) {
    return false;
  }
  if (topology_ == SlidingTopology::Direct) {
    return use < projection.fetch_count;
  }
  const TiledGraphResource *const resource =
      graph_owner_->tiled_graph().resource(uses[use].key.resource);
  return resource != nullptr &&
         resource->kind == GraphResourceKind::ExternalInput &&
         resource->persistence == ResourcePersistence::Backing;
}

bool SlidingInvocation::persist_use(
    const SlidingProjection &projection,
    const std::span<const residency::PageUse> uses,
    const std::size_t use) const noexcept {
  if (!*this || use >= projection.use_count || use >= uses.size() ||
      !writes(uses[use].access)) {
    return false;
  }
  if (topology_ == SlidingTopology::Direct) {
    return use >= projection.fetch_count;
  }
  const TiledGraphResource *const resource =
      graph_owner_->tiled_graph().resource(uses[use].key.resource);
  return resource != nullptr &&
         resource->kind == GraphResourceKind::ExternalOutput;
}

} // namespace rund::compute::detail::residency::execution
