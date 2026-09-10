#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityRelocation() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  // Cross-layout execution searches both C-sized banks but issues only the
  // selected K-prefix. Two pages resident in the opposite order form a real
  // permutation: the Authority must lower it through an evictable arena frame
  // as three dependency-ordered copies, never reject a physically available
  // hit or overwrite one source in place.
  Authority relocated_epoch;
  std::uint32_t a0 = 99u;
  std::uint32_t a1 = 99u;
  std::uint32_t b0 = 99u;
  std::uint32_t b1 = 99u;
  std::uint32_t o0 = 99u;
  if (!relocated_epoch.register_frames(FrameTier::Device, FrameRole::Input,
                                       2u, a0) ||
      !relocated_epoch.register_frames(FrameTier::Device, FrameRole::Input,
                                       2u, a1) ||
      !relocated_epoch.register_frames(FrameTier::Device, FrameRole::Input,
                                       2u, b0) ||
      !relocated_epoch.register_frames(FrameTier::Device, FrameRole::Input,
                                       2u, b1) ||
      !relocated_epoch.register_frames(FrameTier::Device, FrameRole::Output,
                                       2u, o0)) {
    return 74;
  }
  const FrameRegion a_region0{FrameTier::Device, FrameRole::Input, a0, 2u};
  const FrameRegion a_region1{FrameTier::Device, FrameRole::Input, a1, 2u};
  const FrameRegion b_region0{FrameTier::Device, FrameRole::Input, b0, 2u};
  const FrameRegion b_region1{FrameTier::Device, FrameRole::Input, b1, 2u};
  const FrameRegion o_region{FrameTier::Device, FrameRole::Output, o0, 2u};
  const GraphMaterialization a_materialization{
      .resource = 11u,
      .key = {.backing = 91u, .version = 1u, .materialization_hi = 31u},
      .page_bytes = 64u,
      .page_count = 2u,
  };
  const GraphMaterialization b_materialization{
      .resource = 12u,
      .key = {.backing = 92u, .version = 1u, .materialization_hi = 32u},
      .page_bytes = 64u,
      .page_count = 2u,
  };
  const GraphMaterialization o_materialization{
      .resource = 13u,
      .key = {.backing = 93u,
              .version = 1u,
              .materialization_hi = 33u,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 2u,
  };
  std::array<CacheKey, 2u> a_keys{};
  std::array<CacheKey, 2u> b_keys{};
  for (std::uint64_t page = 0u; page < 2u; ++page) {
    if (!project_graph_cache_key(
            a_materialization, PageKey{.resource = 11u, .page = page},
            a_keys[page]) ||
        !project_graph_cache_key(
            b_materialization, PageKey{.resource = 12u, .page = page},
            b_keys[page])) {
      return 75;
    }
  }
  const std::array a_seed{
      CacheUse{.key = a_keys[0], .access = Access::Read},
      CacheUse{.key = a_keys[1], .access = Access::Read},
  };
  const std::array b_seed{
      CacheUse{.key = b_keys[1], .access = Access::Read},
      CacheUse{.key = b_keys[0], .access = Access::Read},
  };
  const AuthorityResult a_seeded = relocated_epoch.begin(
      a_seed, a_region0.tier, a_region0.role, a_region0.first,
      a_region0.count);
  if (!a_seeded || !relocated_epoch.complete(a_seeded.lease.token, true)) {
    return 76;
  }
  const AuthorityResult b_seeded = relocated_epoch.begin(
      b_seed, b_region0.tier, b_region0.role, b_region0.first,
      b_region0.count);
  if (!b_seeded || !relocated_epoch.complete(b_seeded.lease.token, true)) {
    return 76;
  }
  std::array<PageUse, 6u> relocation_uses{};
  for (std::uint32_t page = 0u; page < 2u; ++page) {
    relocation_uses[page] =
        PageUse{.key = {.resource = 11u, .page = page},
                .access = Access::Read,
                .next_use = never,
                .pin = {.first_epoch = 0u, .last_epoch = 0u}};
    relocation_uses[2u + page] =
        PageUse{.key = {.resource = 12u, .page = page},
                .access = Access::Read,
                .next_use = never,
                .pin = {.first_epoch = 0u, .last_epoch = 0u}};
    relocation_uses[4u + page] =
        PageUse{.key = {.resource = 13u, .page = page},
                .access = Access::Write,
                .dirty = {.bytes = 64u},
                .next_use = never,
                .pin = {.first_epoch = 0u, .last_epoch = 0u}};
  }
  const std::array relocation_ports{
      GraphPortRequest{.program_port = 0u,
                       .first_use = 0u,
                       .use_count = 2u,
                       .materialization = a_materialization,
                       .region = a_region0,
                       .cache_regions = {a_region0, a_region1},
                       .cache_region_count = 2u},
      GraphPortRequest{.program_port = 1u,
                       .first_use = 2u,
                       .use_count = 2u,
                       .materialization = b_materialization,
                       .region = b_region0,
                       .cache_regions = {b_region0, b_region1},
                       .cache_region_count = 2u},
      GraphPortRequest{.program_port = 0u,
                       .first_use = 4u,
                       .use_count = 2u,
                       .materialization = o_materialization,
                       .region = o_region,
                       .cache_regions = {o_region},
                       .cache_region_count = 1u},
  };
  const AuthorityResult relocated = relocated_epoch.begin_graph_epoch(
      relocation_uses, relocation_ports, 0u, 0u);
  if (!relocated || relocated.lease.bindings.size() != 6u ||
      relocated.lease.relocations.size() != 3u ||
      relocated.lease.bindings[0].frame != a0 ||
      relocated.lease.bindings[1].frame != a0 + 1u ||
      relocated.lease.bindings[2].frame != b0 ||
      relocated.lease.bindings[3].frame != b0 + 1u ||
      relocated.lease.bindings[2].fetch ||
      relocated.lease.bindings[3].fetch ||
      !relocated.lease.bindings[2].relocated ||
      !relocated.lease.bindings[3].relocated ||
      relocated.lease.relocations.front().target_frame != b1 ||
      !relocated_epoch.activate(relocated.lease.token) ||
      !relocated_epoch.complete(relocated.lease.token, true)) {
    return 77;
  }
  std::array<std::uint8_t, 2u> b_resident{};
  if (!relocated_epoch.probe(b_keys, b_resident, b_region0.first,
                             b_region0.count) ||
      b_resident[0] != 1u || b_resident[1] != 1u) {
    return 78;
  }
  std::array<CacheKey, 2u> o_keys{};
  for (std::uint64_t page = 0u; page < 2u; ++page) {
    if (!project_graph_cache_key(
            o_materialization, PageKey{.resource = 13u, .page = page},
            o_keys[page])) {
      return 79;
    }
  }
  const AuthorityResult o_retire = relocated_epoch.begin_discard(
      o_keys, o_region.first, o_region.count);
  const std::array relocation_regions{a_region0, a_region1, b_region0,
                                      b_region1, o_region};
  if (!o_retire || !relocated_epoch.discard(o_retire.lease.token) ||
      !relocated_epoch.release_frames(relocation_regions)) {
    return 80;
  }

  return 0;
}

} // namespace rund_node_test_pipeline_residency
