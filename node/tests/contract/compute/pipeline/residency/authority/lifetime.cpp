#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityLifetime() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  // Planner-owned transient lifetime is a hard victim boundary. Output bytes
  // remain non-evictable even after the pin interval: only an explicit
  // migration or discard may retire the sole produced copy.
  Authority lifetime;
  std::uint32_t lifetime_base = 99u;
  if (!lifetime.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                lifetime_base)) {
    return 58;
  }
  const FrameRegion lifetime_region{.tier = FrameTier::Device,
                                    .role = FrameRole::Output,
                                    .first = lifetime_base,
                                    .count = 1u};
  const std::array produced{CacheUse{
      .key = {.backing = 70u,
              .version = 1u,
              .page = 0u,
              .domain = CacheDomain::Transient},
      .access = Access::Write,
      .next_use = 2u,
      .dirty = {.offset = 0u, .bytes = 64u},
      .epoch = 0u,
      .retain_until = 2u,
  }};
  const AuthorityResult produced_lease =
      lifetime.begin(produced, lifetime_region.tier, lifetime_region.role,
                     lifetime_region.first, lifetime_region.count);
  if (!produced_lease || !lifetime.activate(produced_lease.lease.token) ||
      !lifetime.complete(produced_lease.lease.token, true)) {
    return 59;
  }
  const std::array early_replacement{CacheUse{
      .key = {.backing = 71u, .version = 1u, .page = 0u},
      .access = Access::Read,
      .next_use = never,
      .epoch = 1u,
      .retain_until = 1u,
  }};
  if (lifetime
          .begin(early_replacement, lifetime_region.tier, lifetime_region.role,
                 lifetime_region.first, lifetime_region.count)
          .failure != AuthorityFailure::Capacity) {
    return 60;
  }
  auto late_replacement = early_replacement;
  late_replacement.front().epoch = 3u;
  late_replacement.front().retain_until = 3u;
  const AuthorityFailure late_failure =
      lifetime
          .begin(late_replacement, lifetime_region.tier, lifetime_region.role,
                 lifetime_region.first, lifetime_region.count)
          .failure;
  const std::array produced_keys{produced.front().key};
  const AuthorityResult retired = lifetime.begin_discard(
      produced_keys, lifetime_region.first, lifetime_region.count);
  if (late_failure != AuthorityFailure::Capacity ||
      !retired || !lifetime.discard(retired.lease.token) ||
      !lifetime.release_frames(std::array{lifetime_region})) {
    return 61;
  }

  Authority transient_miss;
  std::uint32_t transient_base = 99u;
  if (!transient_miss.register_frames(
          FrameTier::Device, FrameRole::Intermediate, 1u, transient_base)) {
    return 62;
  }
  const FrameRegion transient_region{.tier = FrameTier::Device,
                                     .role = FrameRole::Intermediate,
                                     .first = transient_base,
                                     .count = 1u};
  const std::array missing_transient{CacheUse{
      .key = {.backing = 72u,
              .version = 1u,
              .page = 0u,
              .domain = CacheDomain::Transient},
      .access = Access::Read,
      .next_use = never,
      .epoch = 1u,
      .retain_until = 1u,
  }};
  if (transient_miss
              .begin(missing_transient, transient_region.tier,
                    transient_region.role, transient_region.first,
                    transient_region.count)
              .failure != AuthorityFailure::Invalid ||
      !transient_miss.release_frames(std::array{transient_region})) {
    return 63;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
