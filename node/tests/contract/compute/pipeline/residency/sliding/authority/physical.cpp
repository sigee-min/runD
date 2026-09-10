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

[[nodiscard]] int CheckPhysicalForecast() {
  execution::SlidingEvidence evidence{};
  // Bound Direct Forecast has one physical authority. Public model issue is
  // rejected, the opposite-bank future e3 cannot consume the H=1 reservation
  // for e1, and a successful terminal cannot recycle e0's bank0 frame until
  // the callback-return release arrives.
  const std::shared_ptr<const execution::Plan> physical_plan =
      direct_plan(4u, 1u, 1u);
  residency::Authority physical_authority{};
  auto physical_sliding = physical_authority.sliding();
  if (physical_plan == nullptr ||
      !register_direct(physical_authority, 1u, 1u)) {
    return 58;
  }
  const residency::ExecutionLease physical_lease =
      physical_sliding.begin_execution_sliding(*physical_plan);
  execution::Sliding physical_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(physical_plan), physical_lease.token,
      physical_lease.generation, 1u, 1u);
  EpochWork physical0{};
  EpochWork physical2{};
  EpochWork physical3{};
  execution::SlidingTicket bypass{};
  execution::SlidingFetch supply0{};
  execution::SlidingFetch blocked_supply{};
  if (!physical_lease || !physical_run ||
      !physical_sliding.bind_execution_sliding(physical_run) ||
      !physical_run.project(0u, physical0.uses, physical0.projection) ||
      physical_run.issue_fetch(physical0.projection, physical0.uses, 0u,
                               bypass) ||
      !physical_run.project(3u, physical3.uses, physical3.projection) ||
      physical_sliding.issue_execution_sliding_fetch(
          *physical_plan, physical_run, physical3.projection, physical3.uses,
          0u, blocked_supply) ||
      !physical_sliding.issue_execution_sliding_fetch(
          *physical_plan, physical_run, physical0.projection, physical0.uses,
          0u, supply0) ||
      !supply0.requires_backing() || supply0.frame() != 0u ||
      supply0.source() !=
          execution::FetchSource{
              .key = residency::CacheKey{.backing = 11u, .version = 3u},
              .offset = 0u,
              .bytes = 16u,
              .target_offset = 0u,
              .frame_bytes = 16u} ||
      physical_sliding.issue_execution_sliding_fetch(
          *physical_plan, physical_run, physical3.projection, physical3.uses,
          0u, supply0) ||
      !supply0 ||
      !physical_sliding.terminal_execution_sliding_fetch(
          physical_run, supply0, Status::success(),
          execution::TerminalKind::Known, true, supply0.backing_bytes()) ||
      physical_run.quiescent() ||
      !physical_run.project(2u, physical2.uses, physical2.projection) ||
      physical_sliding.issue_execution_sliding_fetch(
          *physical_plan, physical_run, physical2.projection, physical2.uses,
          0u, blocked_supply)) {
    return 59;
  }
  if (!physical_sliding.release_execution_sliding_fetch(physical_run,
                                                          std::move(supply0)) ||
      supply0 ||
      physical_sliding.release_execution_sliding_fetch(physical_run,
                                                         std::move(supply0)) ||
      physical_run.issue_promote(physical0.projection, physical0.uses,
                                 bypass) ||
      !admit_bound(physical_authority, *physical_plan, physical_run,
                   physical0) ||
      !drain_bound(physical_authority, *physical_plan, physical_run, physical0,
                   true)) {
    return 60;
  }
  EpochWork physical1{};
  if (!fetch_bound(physical_authority, *physical_plan, physical_run, 1u,
                   physical1) ||
      !admit_bound(physical_authority, *physical_plan, physical_run,
                   physical1) ||
      !drain_bound(physical_authority, *physical_plan, physical_run, physical1,
                   true) ||
      !fetch_bound(physical_authority, *physical_plan, physical_run, 2u,
                   physical2) ||
      !admit_bound(physical_authority, *physical_plan, physical_run,
                   physical2) ||
      !drain_bound(physical_authority, *physical_plan, physical_run, physical2,
                   true) ||
      !fetch_bound(physical_authority, *physical_plan, physical_run, 3u,
                   physical3) ||
      !admit_bound(physical_authority, *physical_plan, physical_run,
                   physical3) ||
      !drain_bound(physical_authority, *physical_plan, physical_run, physical3,
                   true)) {
    return 61;
  }
  execution::SlidingFinal physical_frozen{};
  residency::ExecutionSlidingFinal physical_prepared{};
  residency::ExecutionSlidingReceipt physical_receipt{};
  if (!physical_run.prepare_final(physical_frozen)) {
    return 621;
  }
  if (!physical_sliding.prepare_execution_sliding(
          *physical_plan, physical_frozen, physical_prepared)) {
    return 622;
  }
  const residency::ExecutionClose physical_closed =
      physical_sliding.accept_execution_sliding(std::move(physical_prepared),
                                                  physical_receipt);
  if (!physical_closed) {
    return 623;
  }
  if (!physical_closed.success) {
    return 624;
  }
  if (!physical_run.accept_final(physical_frozen, std::move(physical_receipt),
                                 evidence)) {
    return 625;
  }
  return 0;
}


[[nodiscard]] int CheckInactiveBank() {
  // Q=1 owns only bank0 for this generation. Keep an unrelated clean row in
  // bank1 and prove Final neither validates nor retires that inactive bank.
  const std::shared_ptr<const execution::Plan> q1_plan =
      direct_plan(1u, 1u, 1u);
  residency::Authority q1_authority{};
  auto q1_sliding = q1_authority.sliding();
  if (q1_plan == nullptr || !register_direct(q1_authority, 1u, 1u)) {
    return 629;
  }
  const residency::CacheKey q1_inactive_key{
      .backing = 99u, .version = 1u, .page = 0u};
  const residency::CacheUse q1_inactive_use{
      .key = q1_inactive_key,
      .access = residency::Access::Read,
      .next_use = residency::NeverUse,
  };
  const residency::AuthorityResult q1_seed = q1_authority.begin(
      std::span<const residency::CacheUse>{&q1_inactive_use, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 1u, 1u);
  std::array<std::uint8_t, 1u> q1_seed_present{0u};
  if (!q1_seed || !q1_authority.complete(q1_seed.lease.token, true) ||
      !q1_authority.probe(
          std::span<const residency::CacheKey>{&q1_inactive_key, 1u},
          q1_seed_present, 1u, 1u) ||
      q1_seed_present[0] != 1u) {
    return 629;
  }
  const residency::ExecutionLease q1_lease =
      q1_sliding.begin_execution_sliding(*q1_plan);
  execution::Sliding q1_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(q1_plan), q1_lease.token,
      q1_lease.generation, 1u, 1u);
  EpochWork q1_work{};
  if (!q1_lease || !q1_run || !q1_sliding.bind_execution_sliding(q1_run) ||
      !fetch_bound(q1_authority, *q1_plan, q1_run, 0u, q1_work) ||
      !admit_bound(q1_authority, *q1_plan, q1_run, q1_work) ||
      !drain_bound(q1_authority, *q1_plan, q1_run, q1_work, true)) {
    return 630;
  }
  execution::SlidingFinal q1_frozen{};
  residency::ExecutionSlidingFinal q1_prepared{};
  residency::ExecutionSlidingReceipt q1_receipt{};
  execution::SlidingEvidence q1_evidence{};
  const residency::ExecutionClose q1_closed =
      q1_run.prepare_final(q1_frozen) && q1_sliding.prepare_execution_sliding(
                                             *q1_plan, q1_frozen, q1_prepared)
          ? q1_sliding.accept_execution_sliding(std::move(q1_prepared),
                                                  q1_receipt)
          : residency::ExecutionClose{};
  if (!q1_closed || !q1_closed.success ||
      !q1_run.accept_final(q1_frozen, std::move(q1_receipt), q1_evidence)) {
    return 631;
  }
  std::array<std::uint8_t, 1u> q1_inactive_present{0u};
  if (!q1_authority.probe(
          std::span<const residency::CacheKey>{&q1_inactive_key, 1u},
          q1_inactive_present, 1u, 1u) ||
      q1_inactive_present[0] != 1u) {
    return 632;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::sliding_authority
