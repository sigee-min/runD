#include "graph.hpp"

#include <kernel/core/checked.hpp>

namespace rund::compute::detail::residency {

bool project_graph_cache_key(const GraphMaterialization &materialization,
                             const PageKey page, CacheKey &key) noexcept {
  key = {};
  if (materialization.resource == 0u ||
      page.resource != materialization.resource ||
      materialization.key.backing == 0u || materialization.key.page != 0u ||
      materialization.page_bytes == 0u || materialization.page_count == 0u ||
      page.page >= materialization.page_count) {
    return false;
  }
  key = materialization.key;
  key.page = page.page;
  key.extent = page.page + 1u == materialization.page_count
                   ? materialization.boundary_extent
                   : 0u;
  return true;
}

[[nodiscard]] bool project_graph_use(const PageUse use,
                                     const GraphMaterialization materialization,
                                     const Access expected,
                                     const std::uint64_t epoch,
                                     CacheUse &projected) noexcept {
  if (use.key.resource != materialization.resource || use.access != expected ||
      (expected == Access::Read ? use.dirty != DirtyRange{}
                                : use.dirty.bytes == 0u) ||
      use.pin.first_epoch > epoch || use.pin.last_epoch < epoch ||
      use.pin.first_epoch > use.pin.last_epoch || use.ready_epoch > epoch ||
      use.prefetch_epoch > use.ready_epoch ||
      use.dirty.offset > materialization.page_bytes ||
      use.dirty.bytes > materialization.page_bytes - use.dirty.offset) {
    return false;
  }
  DirtyExtent dirty{};
  if (use.dirty.bytes != 0u) {
    std::uint64_t page_offset = 0u;
    if (!kernel::checked::mul(use.key.page, materialization.page_bytes,
                              page_offset) ||
        !kernel::checked::add(page_offset, use.dirty.offset, dirty.offset)) {
      return false;
    }
    dirty.bytes = use.dirty.bytes;
  }
  CacheKey key{};
  if (!project_graph_cache_key(materialization, use.key, key)) {
    return false;
  }
  projected = CacheUse{.key = key,
                       .access = use.access,
                       .next_use = use.next_use,
                       .dirty = dirty,
                       .epoch = epoch,
                       .retain_until = use.pin.last_epoch};
  return true;
}

} // namespace rund::compute::detail::residency
