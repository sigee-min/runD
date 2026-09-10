#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"
#include "src/compute/device/residency/registry/view_owner.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityCache() {
  using namespace rund::compute::detail::residency;
  Authority authority;
  std::uint32_t authority_base = 99u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                 authority_base) ||
      authority_base != 0u || authority.frame_capacity() != 2u) {
    return 1;
  }
  const FrameRegion authority_region{.tier = FrameTier::Device,
                                     .role = FrameRole::Input,
                                     .first = authority_base,
                                     .count = 2u};
  const auto never = std::numeric_limits<std::uint64_t>::max();
  const std::array first{
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = 1u},
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 1u},
               .access = Access::Write,
               .next_use = never,
               .dirty = {.offset = 4096u, .bytes = 4096u}},
  };
  const AuthorityResult cold =
      authority.begin(first, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  if (!cold || cold.lease.bindings.size() != 2u ||
      cold.lease.transitions.size() != 3u || cold.lease.bindings[1].fetch ||
      authority
              .begin(first, authority_region.tier, authority_region.role,
                     authority_region.first, authority_region.count)
              .failure != AuthorityFailure::Busy ||
      !authority.complete(cold.lease.token, true)) {
    return 2;
  }
  const std::array second{
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 8u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult replacement =
      authority.begin(second, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  if (!replacement || replacement.lease.transitions.size() != 4u ||
      replacement.lease.transitions[0].kind != TransitionKind::Writeback ||
      replacement.lease.transitions[1].kind != TransitionKind::Unmap ||
      !authority.complete(replacement.lease.token, false)) {
    return 3;
  }
  const AuthorityResult retry =
      authority.begin(second, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  // The failed replacement may have overwritten the victim's physical frame.
  // Rollback therefore invalidates that frame instead of resurrecting stale
  // bytes; retry fetches/maps the new page into the now-empty frame.
  if (!retry || retry.lease.transitions.size() != 2u ||
      !authority.complete(retry.lease.token, true)) {
    return 4;
  }
  const std::array dirty{
      CacheUse{.key = {.backing = 8u, .version = 1u, .page = 0u},
               .access = Access::ReadWrite,
               .next_use = never,
               .dirty = {.offset = 0u, .bytes = 4096u}},
  };
  const AuthorityResult marked =
      authority.begin(dirty, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  if (!marked || !marked.lease.transitions.empty() ||
      !authority.complete(marked.lease.token, true)) {
    return 5;
  }
  const AuthorityResult drain =
      authority.drain_dirty(authority_region.first, authority_region.count);
  if (!drain || drain.lease.bindings.size() != 0u ||
      drain.lease.transitions.size() != 1u ||
      drain.lease.transitions.front().kind != TransitionKind::Writeback ||
      authority
              .begin(dirty, authority_region.tier, authority_region.role,
                     authority_region.first, authority_region.count)
              .failure != AuthorityFailure::Busy ||
      !authority.complete(drain.lease.token, true)) {
    return 6;
  }
  const AuthorityResult clean =
      authority.drain_dirty(authority_region.first, authority_region.count);
  if (!clean || !clean.lease.transitions.empty() ||
      !authority.complete(clean.lease.token, true)) {
    return 7;
  }
  const AuthorityResult dirty_again =
      authority.begin(dirty, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  if (!dirty_again || !authority.complete(dirty_again.lease.token, true)) {
    return 8;
  }
  const AuthorityResult failed_drain =
      authority.drain_dirty(authority_region.first, authority_region.count);
  if (!failed_drain || failed_drain.lease.transitions.size() != 1u ||
      !authority.discard(failed_drain.lease.token)) {
    return 9;
  }
  const std::array third{
      CacheUse{.key = {.backing = 9u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult after_discard =
      authority.begin(third, authority_region.tier, authority_region.role,
                      authority_region.first, authority_region.count);
  if (!after_discard || after_discard.lease.transitions.size() != 2u ||
      after_discard.lease.transitions.front().kind != TransitionKind::Fetch ||
      !authority.complete(after_discard.lease.token, true)) {
    return 10;
  }

  if (!authority.release_frames(std::array{authority_region})) {
    return 10;
  }

  Authority banks;
  std::uint32_t banks_base = 99u;
  if (!banks.register_frames(FrameTier::Device, FrameRole::Input, 4u,
                             banks_base) ||
      banks_base != 0u) {
    return 11;
  }
  const FrameRegion banks_region{.tier = FrameTier::Device,
                                 .role = FrameRole::Input,
                                 .first = banks_base,
                                 .count = 4u};
  const std::array bank_zero{
      CacheUse{.key = {.backing = 10u, .version = 1u, .page = 0u},
               .access = Access::ReadWrite,
               .next_use = never,
               .dirty = {.offset = 0u, .bytes = 4096u}},
      CacheUse{.key = {.backing = 10u, .version = 1u, .page = 1u},
               .access = Access::ReadWrite,
               .next_use = never,
               .dirty = {.offset = 4096u, .bytes = 4096u}},
  };
  const std::array bank_one{
      CacheUse{.key = {.backing = 10u, .version = 1u, .page = 2u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 10u, .version = 1u, .page = 3u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult zero = banks.begin(bank_zero, banks_region.tier,
                                           banks_region.role, banks_base, 2u);
  if (!zero || !banks.activate(zero.lease.token)) {
    return 12;
  }
  const AuthorityResult one = banks.begin(
      bank_one, banks_region.tier, banks_region.role, banks_base + 2u, 2u);
  if (!one || !banks.activate(one.lease.token) ||
      !banks.complete(zero.lease.token, true)) {
    return 13;
  }
  const std::array dirty_keys{bank_zero[0].key, bank_zero[1].key};
  const AuthorityResult page_out =
      banks.begin_writeback(dirty_keys, banks_base, 2u);
  if (!page_out || page_out.lease.transitions.size() != 2u) {
    return 14;
  }
  const std::array replacement_zero{
      CacheUse{.key = {.backing = 11u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 11u, .version = 1u, .page = 1u},
               .access = Access::Read,
               .next_use = never},
  };
  if (banks.begin(replacement_zero, banks_region.tier, banks_region.role,
                  banks_base, 2u)
              .failure != AuthorityFailure::Capacity ||
      !banks.complete(page_out.lease.token, true)) {
    return 15;
  }
  const AuthorityResult refilled = banks.begin(
      replacement_zero, banks_region.tier, banks_region.role, banks_base, 2u);
  if (!refilled || !banks.activate(refilled.lease.token) ||
      !banks.complete(one.lease.token, true) ||
      !banks.complete(refilled.lease.token, true) ||
      !banks.release_frames(std::array{banks_region})) {
    return 16;
  }

  Authority scheduled;
  std::uint32_t scheduled_base = 99u;
  if (!scheduled.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                 scheduled_base) ||
      scheduled_base != 0u) {
    return 17;
  }
  const FrameRegion scheduled_region{.tier = FrameTier::Device,
                                     .role = FrameRole::Input,
                                     .first = scheduled_base,
                                     .count = 2u};
  const std::array scheduled_first{
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 1u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult scheduled_cold = scheduled.begin(
      scheduled_first, scheduled_region.tier, scheduled_region.role,
      scheduled_region.first, scheduled_region.count);
  const StreamPlan linear{2u, 2u, DirtyRange{.bytes = 4096u}, 8192u, 1u};
  const std::array scheduled_regions{scheduled_region};
  const registry_model::ViewCommitPlan view_plan{
      .stream_regions = scheduled_regions,
      .stream = &linear,
      .last = scheduled_first[1].key};
  std::unique_ptr<registry_model::ViewCommitReceipt> view_receipt;
  auto views = scheduled.views();
  if (!scheduled_cold || views.commit_view_plan(view_plan, &view_receipt) ||
      !scheduled.complete(scheduled_cold.lease.token, true) ||
      !views.commit_view_plan(view_plan, &view_receipt) ||
      !views.close_view_commit(view_receipt)) {
    if (view_receipt != nullptr) {
      static_cast<void>(views.quarantine_view_commit(view_receipt));
    }
    return 18;
  }
  const AuthorityResult scheduled_replacement =
      scheduled.begin(third, scheduled_region.tier, scheduled_region.role,
                      scheduled_region.first, scheduled_region.count);
  if (!scheduled_replacement ||
      scheduled_replacement.lease.transitions.size() != 3u ||
      scheduled_replacement.lease.transitions[0].kind !=
          TransitionKind::Unmap ||
      scheduled_replacement.lease.transitions[0].key !=
          scheduled_first[1].key ||
      !scheduled.complete(scheduled_replacement.lease.token, true)) {
    return 19;
  }
  const AuthorityResult uncertain =
      scheduled.begin(third, scheduled_region.tier, scheduled_region.role,
                      scheduled_region.first, scheduled_region.count);
  if (!uncertain || uncertain.lease.bindings.front().fetch ||
      !scheduled.activate(uncertain.lease.token) ||
      !scheduled.complete(uncertain.lease.token, false, true)) {
    return 20;
  }
  const AuthorityResult after_uncertain =
      scheduled.begin(third, scheduled_region.tier, scheduled_region.role,
                      scheduled_region.first, scheduled_region.count);
  if (!after_uncertain || !after_uncertain.lease.bindings.front().fetch ||
      !scheduled.complete(after_uncertain.lease.token, false) ||
      !scheduled.release_frames(std::array{scheduled_region})) {
    return 21;
  }

  Authority global;
  std::uint32_t first_a = 99u;
  std::uint32_t first_b = 99u;
  if (!global.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                              first_a) ||
      first_a != 0u ||
      !global.register_frames(FrameTier::Device, FrameRole::Input, 4u,
                              first_b) ||
      first_b != 2u || global.frame_capacity() != 6u) {
    return 22;
  }
  const FrameRegion global_region_a{.tier = FrameTier::Device,
                                    .role = FrameRole::Input,
                                    .first = first_a,
                                    .count = 2u};
  const FrameRegion global_region_b{.tier = FrameTier::Device,
                                    .role = FrameRole::Input,
                                    .first = first_b,
                                    .count = 4u};
  const std::array region_a{authority_detail::RegionA()};
  const AuthorityResult active_a =
      global.begin(region_a, global_region_a.tier, global_region_a.role,
                   global_region_a.first, global_region_a.count);
  std::uint32_t rejected_base = 99u;
  if (!active_a ||
      global.register_frames(FrameTier::Device, FrameRole::Input, 1u,
                             rejected_base) ||
      rejected_base != 0u || !global.activate(active_a.lease.token) ||
      !global.complete(active_a.lease.token, true)) {
    return 23;
  }
  const std::array region_b{
      CacheUse{.key = {.backing = 21u, .version = 1u, .page = 0u},
               .access = Access::ReadWrite,
               .next_use = never,
               .dirty = {.offset = 0u, .bytes = 4096u}},
  };
  const AuthorityResult dirty_b =
      global.begin(region_b, global_region_b.tier, global_region_b.role,
                   global_region_b.first, global_region_b.count);
  if (!dirty_b || dirty_b.lease.bindings.front().frame < first_b ||
      dirty_b.lease.bindings.front().frame >= first_b + 4u ||
      !global.activate(dirty_b.lease.token) ||
      !global.complete(dirty_b.lease.token, true)) {
    return 24;
  }
  std::array<std::uint8_t, 1u> resident{};
  if (!global.probe(std::array{region_b[0].key}, resident, first_a, 2u) ||
      resident[0] != 0u ||
      !global.probe(std::array{region_b[0].key}, resident, first_b, 4u) ||
      resident[0] != 1u) {
    return 25;
  }
  if (global.release_frames(std::array{global_region_b})) {
    return 26;
  }
  const AuthorityResult scoped_writeback =
      global.begin_writeback(std::array{region_b[0].key}, first_b, 4u);
  if (!scoped_writeback || global.release_frames(std::array{global_region_b}) ||
      !global.complete(scoped_writeback.lease.token, true) ||
      !global.release_frames(std::array{global_region_b})) {
    return 26;
  }
  std::uint32_t reused_base = 99u;
  if (!global.register_frames(FrameTier::Device, FrameRole::Input, 3u,
                              reused_base) ||
      reused_base != first_b ||
      !global.release_frames(std::array{FrameRegion{.tier = FrameTier::Device,
                                                    .role = FrameRole::Input,
                                                    .first = reused_base,
                                                    .count = 3u}}) ||
      !global.release_frames(std::array{global_region_a})) {
    return 27;
  }

  std::uint32_t device_base = 99u;
  std::uint32_t host_base = 99u;
  if (!global.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                              device_base) ||
      !global.register_frames(FrameTier::Host, FrameRole::Input, 2u,
                              host_base) ||
      global.begin(region_a, FrameTier::Device, FrameRole::Input, host_base, 2u)
              .failure != AuthorityFailure::Invalid) {
    return 28;
  }
  const AuthorityResult host =
      global.begin(region_a, FrameTier::Host, FrameRole::Input, host_base, 2u);
  const std::array host_regions{
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = device_base,
                  .count = 2u},
      FrameRegion{.tier = FrameTier::Host,
                  .role = FrameRole::Input,
                  .first = host_base,
                  .count = 2u},
  };
  if (!host || !global.activate(host.lease.token) ||
      global.release_frames(host_regions) ||
      !global.complete(host.lease.token, true) ||
      !global.release_frames(host_regions)) {
    return 29;
  }

  return 0;
}

} // namespace rund_node_test_pipeline_residency
