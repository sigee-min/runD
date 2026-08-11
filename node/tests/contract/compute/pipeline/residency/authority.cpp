#include "local.hpp"

#include "src/compute/device/residency.hpp"

#include <array>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthority() {
  using namespace rund::compute::detail::residency;
  Authority authority;
  if (!authority.configure(4096u, 2u) || !authority.configure(4096u, 2u) ||
      authority.page_bytes() != 4096u || authority.frame_capacity() != 2u ||
      authority.configure(8192u, 2u)) {
    return 1;
  }
  const auto never = std::numeric_limits<std::uint64_t>::max();
  const std::array first{
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = 1u},
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 1u},
               .access = Access::Write,
               .next_use = never},
  };
  const AuthorityResult cold = authority.begin(first);
  if (!cold || cold.lease.bindings.size() != 2u ||
      cold.lease.transitions.size() != 4u ||
      authority.begin(first).failure != AuthorityFailure::Busy ||
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
  const AuthorityResult replacement = authority.begin(second);
  if (!replacement || replacement.lease.transitions.size() != 4u ||
      replacement.lease.transitions[0].kind != TransitionKind::Writeback ||
      replacement.lease.transitions[1].kind != TransitionKind::Unmap ||
      !authority.complete(replacement.lease.token, false)) {
    return 3;
  }
  const AuthorityResult retry = authority.begin(second);
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
               .next_use = never},
  };
  const AuthorityResult marked = authority.begin(dirty);
  if (!marked || !marked.lease.transitions.empty() ||
      !authority.complete(marked.lease.token, true)) {
    return 5;
  }
  const AuthorityResult drain = authority.drain_dirty();
  if (!drain || drain.lease.bindings.size() != 0u ||
      drain.lease.transitions.size() != 1u ||
      drain.lease.transitions.front().kind != TransitionKind::Writeback ||
      authority.begin(dirty).failure != AuthorityFailure::Busy ||
      !authority.complete(drain.lease.token, true)) {
    return 6;
  }
  const AuthorityResult clean = authority.drain_dirty();
  if (!clean || !clean.lease.transitions.empty() ||
      !authority.complete(clean.lease.token, true)) {
    return 7;
  }
  const AuthorityResult dirty_again = authority.begin(dirty);
  if (!dirty_again || !authority.complete(dirty_again.lease.token, true)) {
    return 8;
  }
  const AuthorityResult failed_drain = authority.drain_dirty();
  if (!failed_drain || failed_drain.lease.transitions.size() != 1u ||
      !authority.discard(failed_drain.lease.token)) {
    return 9;
  }
  const std::array third{
      CacheUse{.key = {.backing = 9u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult after_discard = authority.begin(third);
  return after_discard && after_discard.lease.transitions.size() == 2u &&
                 after_discard.lease.transitions.front().kind ==
                     TransitionKind::Fetch &&
                 authority.complete(after_discard.lease.token, true)
             ? 0
             : 10;
}

} // namespace rund_node_test_pipeline_residency
