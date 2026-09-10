#include "local.hpp"

#include <array>
#include <memory>
#include <span>
#include <thread>
#include <utility>

namespace rund_node_test_pipeline_residency::sliding_authority {
namespace {

namespace execution = sliding_detail::execution;
namespace residency = sliding_detail::residency;
using sliding_detail::EpochWork;
using sliding_detail::admit;
using sliding_detail::admit_bound;
using sliding_detail::direct_plan;
using sliding_detail::drain;
using sliding_detail::drain_bound;
using sliding_detail::fetch;
using sliding_detail::fetch_bound;
using sliding_detail::register_direct;
using sliding_detail::region;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::Type;

} // namespace

[[nodiscard]] int CheckCacheAuthority() {
  execution::SlidingEvidence evidence{};
  const std::shared_ptr<const execution::Plan> malformed_plan =
      direct_plan(1u, 1u, 1u);
  if (malformed_plan == nullptr) {
    return 69;
  }
  // A clean inherited row with a different key is still an Authority-owned
  // victim. The physical journal does not require pre-run cache metadata to
  // have been mirrored into a Sliding InputCell.
  residency::Authority inherited_authority{};
  auto inherited_sliding = inherited_authority.sliding();
  if (!register_direct(inherited_authority, 1u, 1u)) {
    return 69;
  }
  const residency::CacheUse inherited_key{
      .key = {.backing = 99u, .version = 1u, .page = 0u},
      .access = residency::Access::Read,
      .next_use = residency::NeverUse,
  };
  const residency::AuthorityResult inherited_seed = inherited_authority.begin(
      std::span<const residency::CacheUse>{&inherited_key, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 0u, 1u);
  const residency::ExecutionLease inherited_lease =
      inherited_seed &&
              inherited_authority.complete(inherited_seed.lease.token, true)
          ? inherited_sliding.begin_execution_sliding(*malformed_plan)
          : residency::ExecutionLease{};
  execution::Sliding inherited_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(malformed_plan),
      inherited_lease.token, inherited_lease.generation, 1u, 1u);
  EpochWork inherited_work{};
  execution::SlidingFetch inherited_fetch{};
  if (!inherited_lease || !inherited_run ||
      !inherited_sliding.bind_execution_sliding(inherited_run) ||
      !inherited_run.project(0u, inherited_work.uses,
                             inherited_work.projection) ||
      !inherited_sliding.issue_execution_sliding_fetch(
          *malformed_plan, inherited_run, inherited_work.projection,
          inherited_work.uses, 0u, inherited_fetch) ||
      !inherited_fetch.requires_backing() || inherited_fetch.frame() != 0u ||
      !inherited_sliding.terminal_execution_sliding_fetch(
          inherited_run, inherited_fetch, Status::success(),
          execution::TerminalKind::Known, true,
          inherited_fetch.backing_bytes()) ||
      !inherited_sliding.release_execution_sliding_fetch(
          inherited_run, std::move(inherited_fetch)) ||
      !admit_bound(inherited_authority, *malformed_plan, inherited_run,
                   inherited_work) ||
      !drain_bound(inherited_authority, *malformed_plan, inherited_run,
                   inherited_work, true)) {
    return 70;
  }
  execution::SlidingFinal inherited_frozen{};
  residency::ExecutionSlidingFinal inherited_prepared{};
  residency::ExecutionSlidingReceipt inherited_receipt{};
  if (!inherited_run.prepare_final(inherited_frozen) ||
      !inherited_sliding.prepare_execution_sliding(
          *malformed_plan, inherited_frozen, inherited_prepared) ||
      !inherited_sliding.accept_execution_sliding(
          std::move(inherited_prepared), inherited_receipt) ||
      !inherited_run.accept_final(inherited_frozen,
                                  std::move(inherited_receipt), evidence) ||
      evidence.fetch_calls != 1u || evidence.fetch_hits != 0u) {
    return 71;
  }

  // A clean pre-run physical row is an Authority-authenticated hit. It names
  // the real Host frame and creates no backing terminal surface.
  residency::Authority hit_authority{};
  auto hit_sliding = hit_authority.sliding();
  if (!register_direct(hit_authority, 1u, 1u)) {
    return 66;
  }
  const residency::CacheUse cached_input{
      .key = {.backing = 11u, .version = 3u, .page = 0u},
      .access = residency::Access::Read,
      .next_use = 1u,
  };
  const residency::AuthorityResult cached = hit_authority.begin(
      std::span<const residency::CacheUse>{&cached_input, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 0u, 1u);
  const residency::ExecutionLease hit_lease =
      cached && hit_authority.complete(cached.lease.token, true)
          ? hit_sliding.begin_execution_sliding(*malformed_plan)
          : residency::ExecutionLease{};
  execution::Sliding hit_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(malformed_plan), hit_lease.token,
      hit_lease.generation, 1u, 1u);
  EpochWork hit_work{};
  execution::SlidingFetch hit_fetch{};
  execution::SlidingFinal hit_frozen{};
  if (!hit_lease || !hit_run ||
      !hit_sliding.bind_execution_sliding(hit_run) ||
      !hit_run.project(0u, hit_work.uses, hit_work.projection) ||
      !hit_sliding.issue_execution_sliding_fetch(
          *malformed_plan, hit_run, hit_work.projection, hit_work.uses, 0u,
          hit_fetch) ||
      hit_fetch.requires_backing() || hit_fetch.frame() != 0u ||
      hit_sliding.terminal_execution_sliding_fetch(
          hit_run, hit_fetch, Status::success(), execution::TerminalKind::Known,
          true, hit_fetch.source().bytes) ||
      hit_run.prepare_final(hit_frozen) ||
      !hit_sliding.release_execution_sliding_fetch(hit_run,
                                                     std::move(hit_fetch)) ||
      hit_fetch ||
      !admit_bound(hit_authority, *malformed_plan, hit_run, hit_work) ||
      !drain_bound(hit_authority, *malformed_plan, hit_run, hit_work, true)) {
    return 67;
  }
  residency::ExecutionSlidingFinal hit_prepared{};
  residency::ExecutionSlidingReceipt hit_receipt{};
  if (!hit_run.prepare_final(hit_frozen) ||
      !hit_sliding.prepare_execution_sliding(*malformed_plan, hit_frozen,
                                               hit_prepared) ||
      !hit_sliding.accept_execution_sliding(std::move(hit_prepared),
                                              hit_receipt) ||
      !hit_run.accept_final(hit_frozen, std::move(hit_receipt), evidence) ||
      evidence.fetch_calls != 0u || evidence.fetch_hits != 1u) {
    return 68;
  }

  // A future bank-frontier hit stays releasable when an earlier miss fails.
  // Failure freezes new work but cannot erase an already-issued physical
  // handoff before its exact consumer returns.
  const std::shared_ptr<const execution::Plan> handoff_plan =
      direct_plan(2u, 1u, 1u);
  residency::Authority handoff_authority{};
  auto handoff_sliding = handoff_authority.sliding();
  if (handoff_plan == nullptr || !register_direct(handoff_authority, 1u, 1u)) {
    return 72;
  }
  const residency::CacheUse bank1_hit{
      .key = {.backing = 11u, .version = 3u, .page = 1u},
      .access = residency::Access::Read,
      .next_use = 3u,
  };
  const residency::AuthorityResult bank1_seed = handoff_authority.begin(
      std::span<const residency::CacheUse>{&bank1_hit, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 1u, 1u);
  const residency::ExecutionLease handoff_lease =
      bank1_seed && handoff_authority.complete(bank1_seed.lease.token, true)
          ? handoff_sliding.begin_execution_sliding(*handoff_plan)
          : residency::ExecutionLease{};
  execution::Sliding handoff_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(handoff_plan), handoff_lease.token,
      handoff_lease.generation, 1u, 1u);
  EpochWork handoff0{};
  EpochWork handoff1{};
  execution::SlidingFetch miss0{};
  execution::SlidingFetch hit1{};
  if (!handoff_lease || !handoff_run ||
      !handoff_sliding.bind_execution_sliding(handoff_run) ||
      !handoff_run.project(1u, handoff1.uses, handoff1.projection) ||
      !handoff_sliding.issue_execution_sliding_fetch(
          *handoff_plan, handoff_run, handoff1.projection, handoff1.uses, 0u,
          hit1) ||
      hit1.requires_backing() ||
      !handoff_run.project(0u, handoff0.uses, handoff0.projection) ||
      !handoff_sliding.issue_execution_sliding_fetch(
          *handoff_plan, handoff_run, handoff0.projection, handoff0.uses, 0u,
          miss0) ||
      !miss0.requires_backing() ||
      !handoff_sliding.terminal_execution_sliding_fetch(
          handoff_run, miss0, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::Known, false, 0u) ||
      !handoff_sliding.release_execution_sliding_fetch(handoff_run,
                                                         std::move(miss0)) ||
      !handoff_sliding.release_execution_sliding_fetch(handoff_run,
                                                         std::move(hit1))) {
    return 73;
  }
  execution::SlidingFinal handoff_frozen{};
  residency::ExecutionSlidingFinal handoff_prepared{};
  residency::ExecutionSlidingReceipt handoff_receipt{};
  if (!handoff_run.prepare_final(handoff_frozen) ||
      !handoff_sliding.prepare_execution_sliding(
          *handoff_plan, handoff_frozen, handoff_prepared) ||
      !handoff_sliding.accept_execution_sliding(std::move(handoff_prepared),
                                                  handoff_receipt) ||
      !handoff_run.accept_final(handoff_frozen, std::move(handoff_receipt),
                                evidence) ||
      !evidence.has_failure || evidence.fetch_calls != 1u ||
      evidence.fetch_hits != 1u) {
    return 74;
  }

  // Unknown keeps the backend's exact failure reason while raising the
  // run-wide certainty to UnknownMayWrite. It cannot enter Known 2PC and its
  // retained physical handoff remains owned by quarantine.
  residency::Authority unknown_authority{};
  auto unknown_sliding = unknown_authority.sliding();
  if (!register_direct(unknown_authority, 1u, 1u)) {
    return 75;
  }
  const residency::ExecutionLease unknown_lease =
      unknown_sliding.begin_execution_sliding(*malformed_plan);
  execution::Sliding unknown_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(malformed_plan), unknown_lease.token,
      unknown_lease.generation, 1u, 1u);
  EpochWork unknown_work{};
  execution::SlidingFetch unknown_fetch{};
  execution::SlidingFinal unknown_frozen{};
  if (!unknown_lease || !unknown_run ||
      !unknown_sliding.bind_execution_sliding(unknown_run) ||
      !unknown_run.project(0u, unknown_work.uses, unknown_work.projection) ||
      !unknown_sliding.issue_execution_sliding_fetch(
          *malformed_plan, unknown_run, unknown_work.projection,
          unknown_work.uses, 0u, unknown_fetch) ||
      !unknown_sliding.terminal_execution_sliding_fetch(
          unknown_run, unknown_fetch, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::UnknownMayWrite, false, 0u) ||
      !unknown_sliding.release_execution_sliding_fetch(
          unknown_run, std::move(unknown_fetch)) ||
      !unknown_run.snapshot(evidence) || !evidence.quarantined ||
      evidence.status.reason() != Reason::BackendFailed ||
      evidence.terminal != execution::TerminalKind::UnknownMayWrite ||
      !evidence.first_failure_may_write ||
      unknown_run.prepare_final(unknown_frozen)) {
    return 76;
  }

  // An unconsumed future row is scored by its immediate consumer, not its
  // post-consumption next-use. With H=2, e4 cannot thrash away the already
  // fetched e2 while e0 is still the bank frontier.
  const std::shared_ptr<const execution::Plan> forecast_plan =
      direct_plan(5u, 1u, 2u);
  residency::Authority forecast_authority{};
  auto forecast_sliding = forecast_authority.sliding();
  if (forecast_plan == nullptr ||
      !register_direct(forecast_authority, 1u, 2u)) {
    return 77;
  }
  const residency::ExecutionLease forecast_lease =
      forecast_sliding.begin_execution_sliding(*forecast_plan);
  execution::Sliding forecast_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(forecast_plan), forecast_lease.token,
      forecast_lease.generation, 2u, 2u);
  std::array<EpochWork, 5u> forecast_work{};
  execution::SlidingFetch forecast0{};
  execution::SlidingFetch forecast2{};
  execution::SlidingFetch premature4{};
  if (!forecast_lease || !forecast_run ||
      !forecast_sliding.bind_execution_sliding(forecast_run) ||
      !forecast_run.project(0u, forecast_work[0].uses,
                            forecast_work[0].projection) ||
      !forecast_sliding.issue_execution_sliding_fetch(
          *forecast_plan, forecast_run, forecast_work[0].projection,
          forecast_work[0].uses, 0u, forecast0) ||
      !forecast_sliding.terminal_execution_sliding_fetch(
          forecast_run, forecast0, Status::success(),
          execution::TerminalKind::Known, true, forecast0.backing_bytes()) ||
      !forecast_sliding.release_execution_sliding_fetch(
          forecast_run, std::move(forecast0)) ||
      !forecast_run.project(2u, forecast_work[2].uses,
                            forecast_work[2].projection) ||
      !forecast_sliding.issue_execution_sliding_fetch(
          *forecast_plan, forecast_run, forecast_work[2].projection,
          forecast_work[2].uses, 0u, forecast2) ||
      !forecast_sliding.terminal_execution_sliding_fetch(
          forecast_run, forecast2, Status::success(),
          execution::TerminalKind::Known, true, forecast2.backing_bytes()) ||
      !forecast_sliding.release_execution_sliding_fetch(
          forecast_run, std::move(forecast2)) ||
      !forecast_run.project(4u, forecast_work[4].uses,
                            forecast_work[4].projection) ||
      forecast_sliding.issue_execution_sliding_fetch(
          *forecast_plan, forecast_run, forecast_work[4].projection,
          forecast_work[4].uses, 0u, premature4)) {
    return 78;
  }
  if (!admit_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[0]) ||
      !drain_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[0], true) ||
      !fetch_bound(forecast_authority, *forecast_plan, forecast_run, 1u,
                   forecast_work[1]) ||
      !admit_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[1]) ||
      !drain_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[1], true) ||
      !admit_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[2]) ||
      !drain_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[2], true) ||
      !fetch_bound(forecast_authority, *forecast_plan, forecast_run, 3u,
                   forecast_work[3]) ||
      !admit_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[3]) ||
      !drain_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[3], true) ||
      !fetch_bound(forecast_authority, *forecast_plan, forecast_run, 4u,
                   forecast_work[4]) ||
      !admit_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[4]) ||
      !drain_bound(forecast_authority, *forecast_plan, forecast_run,
                   forecast_work[4], true)) {
    return 79;
  }
  execution::SlidingFinal forecast_frozen{};
  residency::ExecutionSlidingFinal forecast_prepared{};
  residency::ExecutionSlidingReceipt forecast_receipt{};
  if (!forecast_run.prepare_final(forecast_frozen) ||
      !forecast_sliding.prepare_execution_sliding(
          *forecast_plan, forecast_frozen, forecast_prepared) ||
      !forecast_sliding.accept_execution_sliding(std::move(forecast_prepared),
                                                   forecast_receipt) ||
      !forecast_run.accept_final(forecast_frozen, std::move(forecast_receipt),
                                 evidence)) {
    return 80;
  }

  // A finite Direct retain_until is run-wide. Bank0 needs rows for e0 and e2,
  // so H=1 is an impossible success run, not recoverable backpressure.
  const std::shared_ptr<const execution::Plan> retained_plan =
      direct_plan(3u, 1u, 1u, 2u);
  residency::Authority retained_authority{};
  auto retained_sliding = retained_authority.sliding();
  if (retained_plan == nullptr ||
      !register_direct(retained_authority, 1u, 1u)) {
    return 81;
  }
  const residency::ExecutionLease retained_lease =
      retained_sliding.begin_execution_sliding(*retained_plan);
  execution::Sliding retained_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(retained_plan), retained_lease.token,
      retained_lease.generation, 1u, 1u);
  if (!retained_lease || retained_run ||
      !retained_sliding.abandon_execution_sliding(*retained_plan,
                                                    retained_lease)) {
    return 82;
  }

  // Plan can project incomplete boundary/tail sources, but a bound physical
  // run must fail before owner bind until the fill/finalize service exists.
  const std::shared_ptr<const execution::Plan> short_plan =
      direct_plan(2u, 1u, 1u, residency::NeverUse, 31u);
  residency::Authority short_authority{};
  auto short_sliding = short_authority.sliding();
  if (short_plan == nullptr || !register_direct(short_authority, 1u, 1u)) {
    return 83;
  }
  const residency::ExecutionLease short_lease =
      short_sliding.begin_execution_sliding(*short_plan);
  execution::Sliding short_control = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(short_plan), short_lease.token,
      short_lease.generation, 1u, 1u);
  if (!short_lease || short_control ||
      !short_sliding.abandon_execution_sliding(*short_plan, short_lease)) {
    return 83;
  }
  const std::shared_ptr<const execution::Plan> short_zero_plan =
      direct_plan(2u, 1u, 1u, residency::NeverUse, 31u,
                  execution::FetchFill::ZeroInactiveTail);
  residency::Authority short_zero_authority{};
  auto short_zero_sliding = short_zero_authority.sliding();
  if (short_zero_plan == nullptr ||
      !register_direct(short_zero_authority, 1u, 1u)) {
    return 83;
  }
  const residency::ExecutionLease short_zero_lease =
      short_zero_sliding.begin_execution_sliding(*short_zero_plan);
  execution::Sliding short_zero_control = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(short_zero_plan),
      short_zero_lease.token, short_zero_lease.generation, 1u, 1u);
  if (!short_zero_lease || !short_zero_control ||
      !short_zero_sliding.abandon_execution_sliding(*short_zero_plan,
                                                      short_zero_lease)) {
    return 83;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::sliding_authority
