#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"
#include "src/compute/device/residency/registry/view_owner.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityViews() {
  using namespace rund::compute::detail::residency;
  // One raw extent may expose incompatible frame geometries, but only one
  // semantic view can own valid cache rows at a time. Switching views evicts
  // every clean row atomically; a dirty row blocks the switch until its
  // explicit terminal path proves where the newer bytes went.
  Authority aliased;
  std::uint32_t canonical_first = 0u;
  std::uint32_t reblocked_first = 0u;
  if (!aliased.register_frames(FrameTier::Device, FrameRole::Input, 2u, 701u,
                               801u, canonical_first) ||
      !aliased.register_frames(FrameTier::Device, FrameRole::Output, 4u, 701u,
                               802u, reblocked_first)) {
    return 81;
  }
  const FrameRegion canonical_region{FrameTier::Device, FrameRole::Input,
                                     canonical_first, 2u};
  const FrameRegion reblocked_region{FrameTier::Device, FrameRole::Output,
                                     reblocked_first, 4u};
  const std::array canonical_keys{
      CacheKey{.backing = 301u, .version = 1u, .page = 0u},
      CacheKey{.backing = 301u, .version = 1u, .page = 1u},
  };
  const std::array canonical_uses{
      CacheUse{.key = canonical_keys[0], .access = Access::Read},
      CacheUse{.key = canonical_keys[1], .access = Access::Read},
  };
  const ViewActivationResult selected_canonical =
      aliased.views().activate_views(std::array{canonical_region});
  const AuthorityResult canonical_seed = aliased.begin(
      canonical_uses, canonical_region.tier, canonical_region.role,
      canonical_region.first, canonical_region.count);
  if (!selected_canonical || selected_canonical.eviction_count != 0u ||
      !canonical_seed || !aliased.complete(canonical_seed.lease.token, true)) {
    return 82;
  }
  const ViewActivationResult selected_reblocked =
      aliased.views().activate_views(std::array{reblocked_region});
  std::array<std::uint8_t, 2u> canonical_resident{};
  if (!selected_reblocked || selected_reblocked.eviction_count != 2u ||
      !aliased.probe(canonical_keys, canonical_resident, canonical_region.first,
                     canonical_region.count) ||
      canonical_resident[0] != 0u || canonical_resident[1] != 0u) {
    return 83;
  }
  const CacheKey dirty_key{.backing = 302u, .version = 1u, .page = 0u};
  const std::array dirty_use{CacheUse{
      .key = dirty_key,
      .access = Access::Write,
      .dirty = {.offset = 0u, .bytes = 32u},
  }};
  const AuthorityResult dirtied =
      aliased.begin(dirty_use, reblocked_region.tier, reblocked_region.role,
                    reblocked_region.first, reblocked_region.count);
  if (!dirtied || !aliased.complete(dirtied.lease.token, true)) {
    return 84;
  }
  const ViewActivationResult rejected_dirty =
      aliased.views().activate_views(std::array{canonical_region});
  const AuthorityResult dirty_drain = aliased.begin_writeback(
      std::array{dirty_key}, reblocked_region.first, reblocked_region.count);
  if (rejected_dirty.failure != AuthorityFailure::Capacity || !dirty_drain ||
      !aliased.discard(dirty_drain.lease.token)) {
    return 85;
  }
  const ViewActivationResult selected_after_discard =
      aliased.views().activate_views(std::array{canonical_region});
  const std::array alias_regions{canonical_region, reblocked_region};
  if (!selected_after_discard || selected_after_discard.eviction_count != 0u ||
      !aliased.release_frames(alias_regions)) {
    return 86;
  }

  return 0;
}

} // namespace rund_node_test_pipeline_residency
