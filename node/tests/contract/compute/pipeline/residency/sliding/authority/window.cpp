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

[[nodiscard]] int CheckHaloReuse() {
  execution::SlidingEvidence evidence{};
  // A Direct halo miss authenticates the exact prior Resident Host frame and
  // copies only its overlap. That source remains physically Pinned until the
  // current Fetch callback returns, so a farther same-bank issue cannot
  // overwrite bytes still being copied.
  const std::shared_ptr<const execution::Plan> overlap_plan =
      direct_plan(4u, 1u, 1u, residency::NeverUse, 61u,
                  execution::FetchFill::RepeatBoundary, 24u, 16u, 4u, 4u, 1u);
  residency::Authority reuse_authority{};
  auto reuse_sliding = reuse_authority.sliding();
  if (overlap_plan == nullptr || !register_direct(reuse_authority, 1u, 1u)) {
    return 80;
  }
  const residency::ExecutionLease reuse_lease =
      reuse_sliding.begin_execution_sliding(*overlap_plan);
  execution::Sliding reuse_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(overlap_plan), reuse_lease.token,
      reuse_lease.generation, 1u, 1u);
  std::array<EpochWork, 4u> reuse_work{};
  execution::SlidingFetch reuse_fetch{};
  if (!reuse_lease || !reuse_run ||
      !reuse_sliding.bind_execution_sliding(reuse_run) ||
      !reuse_run.project(0u, reuse_work[0].uses, reuse_work[0].projection) ||
      !reuse_sliding.issue_execution_sliding_fetch(
          *overlap_plan, reuse_run, reuse_work[0].projection,
          reuse_work[0].uses, 0u, reuse_fetch) ||
      !reuse_fetch.requires_backing() || reuse_fetch.reuses_frame() ||
      reuse_fetch.backing_bytes() != 20u ||
      !reuse_sliding.terminal_execution_sliding_fetch(
          reuse_run, reuse_fetch, Status::success(),
          execution::TerminalKind::Known, true, reuse_fetch.backing_bytes()) ||
      !reuse_sliding.release_execution_sliding_fetch(
          reuse_run, std::move(reuse_fetch)) ||
      !admit_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[0]) ||
      !drain_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[0],
                   true)) {
    return 81;
  }
  if (!reuse_run.project(1u, reuse_work[1].uses, reuse_work[1].projection) ||
      !reuse_sliding.issue_execution_sliding_fetch(
          *overlap_plan, reuse_run, reuse_work[1].projection,
          reuse_work[1].uses, 0u, reuse_fetch) ||
      !reuse_fetch.requires_backing() || !reuse_fetch.reuses_frame() ||
      reuse_fetch.frame() != 1u || reuse_fetch.reuse_frame() != 0u ||
      reuse_fetch.backing_bytes() != 16u ||
      reuse_fetch.reuse().key.page != 0u || reuse_fetch.reuse().bytes != 8u ||
      !reuse_sliding.terminal_execution_sliding_fetch(
          reuse_run, reuse_fetch, Status::success(),
          execution::TerminalKind::Known, true, reuse_fetch.backing_bytes()) ||
      !reuse_sliding.release_execution_sliding_fetch(
          reuse_run, std::move(reuse_fetch)) ||
      !admit_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[1]) ||
      !drain_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[1],
                   true)) {
    return 82;
  }
  execution::SlidingFetch pinned_fetch{};
  execution::SlidingFetch blocked_reuse{};
  if (!reuse_run.project(2u, reuse_work[2].uses, reuse_work[2].projection) ||
      !reuse_sliding.issue_execution_sliding_fetch(
          *overlap_plan, reuse_run, reuse_work[2].projection,
          reuse_work[2].uses, 0u, pinned_fetch) ||
      !pinned_fetch.reuses_frame() || pinned_fetch.frame() != 0u ||
      pinned_fetch.reuse_frame() != 1u || pinned_fetch.backing_bytes() != 16u ||
      !reuse_sliding.terminal_execution_sliding_fetch(
          reuse_run, pinned_fetch, Status::success(),
          execution::TerminalKind::Known, true, pinned_fetch.backing_bytes()) ||
      !reuse_run.project(3u, reuse_work[3].uses, reuse_work[3].projection) ||
      reuse_sliding.issue_execution_sliding_fetch(
          *overlap_plan, reuse_run, reuse_work[3].projection,
          reuse_work[3].uses, 0u, blocked_reuse) ||
      !reuse_sliding.release_execution_sliding_fetch(
          reuse_run, std::move(pinned_fetch)) ||
      !admit_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[2]) ||
      !drain_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[2],
                   true)) {
    return 83;
  }
  if (!reuse_sliding.issue_execution_sliding_fetch(
          *overlap_plan, reuse_run, reuse_work[3].projection,
          reuse_work[3].uses, 0u, reuse_fetch) ||
      !reuse_fetch.reuses_frame() || reuse_fetch.frame() != 1u ||
      reuse_fetch.reuse_frame() != 0u || reuse_fetch.backing_bytes() != 9u ||
      !reuse_sliding.terminal_execution_sliding_fetch(
          reuse_run, reuse_fetch, Status::success(),
          execution::TerminalKind::Known, true, reuse_fetch.backing_bytes()) ||
      !reuse_sliding.release_execution_sliding_fetch(
          reuse_run, std::move(reuse_fetch)) ||
      !admit_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[3]) ||
      !drain_bound(reuse_authority, *overlap_plan, reuse_run, reuse_work[3],
                   true)) {
    return 84;
  }
  execution::SlidingFinal reuse_frozen{};
  residency::ExecutionSlidingFinal reuse_prepared{};
  residency::ExecutionSlidingReceipt reuse_receipt{};
  if (!reuse_run.prepare_final(reuse_frozen) ||
      !reuse_sliding.prepare_execution_sliding(*overlap_plan, reuse_frozen,
                                                 reuse_prepared) ||
      !reuse_sliding.accept_execution_sliding(std::move(reuse_prepared),
                                                reuse_receipt) ||
      !reuse_run.accept_final(reuse_frozen, std::move(reuse_receipt),
                              evidence) ||
      evidence.fetch_bytes != 61u) {
    return 85;
  }
  return 0;
}


[[nodiscard]] int CheckOutputReservation() {
  execution::SlidingEvidence evidence{};
  // K=2 proves physical Promote reserves distinct Device/Host output rows and
  // distinct logical OutputCells before either native output can mutate.
  const std::shared_ptr<const execution::Plan> wide_plan =
      direct_plan(2u, 2u, 2u);
  residency::Authority wide_authority{};
  auto wide_sliding = wide_authority.sliding();
  if (wide_plan == nullptr || !register_direct(wide_authority, 2u, 2u)) {
    return 626;
  }
  const residency::ExecutionLease wide_lease =
      wide_sliding.begin_execution_sliding(*wide_plan);
  execution::Sliding wide_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(wide_plan), wide_lease.token,
      wide_lease.generation, 2u, 2u);
  EpochWork wide_work{};
  execution::SlidingPromote wide_promote{};
  if (!wide_lease || !wide_run ||
      !wide_sliding.bind_execution_sliding(wide_run) ||
      !fetch_bound(wide_authority, *wide_plan, wide_run, 0u, wide_work) ||
      !wide_sliding.issue_execution_sliding_promote(
          *wide_plan, wide_run, wide_work.projection, wide_work.uses,
          wide_promote) ||
      wide_promote.host_frames().size() != 2u ||
      wide_promote.device_input_frames().size() != 2u ||
      wide_promote.device_output_frames().size() != 2u ||
      wide_promote.host_output_frames().size() != 2u ||
      wide_promote.device_output_frames()[0u] ==
          wide_promote.device_output_frames()[1u] ||
      wide_promote.host_output_frames()[0u] ==
          wide_promote.host_output_frames()[1u] ||
      wide_promote.expected_bytes() != 32u ||
      !wide_sliding.terminal_execution_sliding_promote(
          wide_run, wide_promote, Status::success(),
          execution::TerminalKind::Known, true, 32u) ||
      !wide_sliding.release_execution_sliding_promote(
          wide_run, std::move(wide_promote)) ||
      !wide_sliding.issue_execution_sliding_native(
          *wide_plan, wide_run, wide_work.projection, wide_work.uses,
          *(wide_work.physical_native =
                std::make_shared<execution::SlidingNative>())) ||
      !drain_bound(wide_authority, *wide_plan, wide_run, wide_work, true)) {
    return 627;
  }
  execution::SlidingFinal wide_frozen{};
  residency::ExecutionSlidingFinal wide_prepared{};
  residency::ExecutionSlidingReceipt wide_receipt{};
  if (!wide_run.prepare_final(wide_frozen) ||
      !wide_sliding.prepare_execution_sliding(*wide_plan, wide_frozen,
                                                wide_prepared) ||
      !wide_sliding.accept_execution_sliding(std::move(wide_prepared),
                                               wide_receipt) ||
      !wide_run.accept_final(wide_frozen, std::move(wide_receipt), evidence)) {
    return 628;
  }
  return 0;
}


[[nodiscard]] int CheckMalformedReceipts() {
  execution::SlidingEvidence evidence{};
  // A contradictory successful receipt with a short byte count cannot trust
  // caller may_write=false. It remains Completing until release, becomes an
  // exact Known may-write failure, contributes no authenticated byte count,
  // and closes only through the whole-generation Authority abort.
  const std::shared_ptr<const execution::Plan> malformed_plan =
      direct_plan(1u, 1u, 1u);
  residency::Authority malformed_authority{};
  auto malformed_sliding = malformed_authority.sliding();
  if (malformed_plan == nullptr ||
      !register_direct(malformed_authority, 1u, 1u)) {
    return 63;
  }
  const residency::ExecutionLease malformed_lease =
      malformed_sliding.begin_execution_sliding(*malformed_plan);
  execution::Sliding malformed_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(malformed_plan),
      malformed_lease.token, malformed_lease.generation, 1u, 1u);
  EpochWork physical_malformed_work{};
  execution::SlidingFetch physical_malformed_fetch{};
  execution::SlidingFinal malformed_frozen{};
  residency::ExecutionSlidingFinal malformed_prepared{};
  residency::ExecutionSlidingReceipt malformed_receipt{};
  if (!malformed_lease || !malformed_run ||
      !malformed_sliding.bind_execution_sliding(malformed_run) ||
      !malformed_run.project(0u, physical_malformed_work.uses,
                             physical_malformed_work.projection) ||
      !malformed_sliding.issue_execution_sliding_fetch(
          *malformed_plan, malformed_run, physical_malformed_work.projection,
          physical_malformed_work.uses, 0u, physical_malformed_fetch) ||
      !physical_malformed_fetch.requires_backing() ||
      !malformed_sliding.terminal_execution_sliding_fetch(
          malformed_run, physical_malformed_fetch, Status::success(),
          execution::TerminalKind::Known, false,
          physical_malformed_fetch.backing_bytes() - 1u) ||
      malformed_run.prepare_final(malformed_frozen) ||
      !malformed_sliding.release_execution_sliding_fetch(
          malformed_run, std::move(physical_malformed_fetch)) ||
      !malformed_run.prepare_final(malformed_frozen) ||
      !malformed_sliding.prepare_execution_sliding(
          *malformed_plan, malformed_frozen, malformed_prepared)) {
    return 64;
  }
  const residency::ExecutionClose malformed_closed =
      malformed_sliding.accept_execution_sliding(
          std::move(malformed_prepared), malformed_receipt);
  if (!malformed_closed || malformed_closed.success || !malformed_receipt ||
      !malformed_run.accept_final(malformed_frozen,
                                  std::move(malformed_receipt), evidence) ||
      !evidence.has_failure || !evidence.first_failure_may_write ||
      evidence.fetch_calls != 1u || evidence.fetch_bytes != 0u) {
    return 65;
  }

  // A Known failed Promote that reports a nonzero partial byte count cannot
  // claim no-write. Authority invalidates its exact DeviceInput Mapping during
  // callback-return release and the generation closes as may-write failure.
  residency::Authority partial_promote_authority{};
  auto partial_promote_sliding = partial_promote_authority.sliding();
  if (!register_direct(partial_promote_authority, 1u, 1u)) {
    return 65;
  }
  const residency::ExecutionLease partial_promote_lease =
      partial_promote_sliding.begin_execution_sliding(*malformed_plan);
  execution::Sliding partial_promote_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(malformed_plan),
      partial_promote_lease.token, partial_promote_lease.generation, 1u, 1u);
  EpochWork partial_promote_work{};
  execution::SlidingPromote partial_promote{};
  execution::SlidingFinal partial_promote_frozen{};
  residency::ExecutionSlidingFinal partial_promote_prepared{};
  residency::ExecutionSlidingReceipt partial_promote_receipt{};
  if (!partial_promote_lease || !partial_promote_run ||
      !partial_promote_sliding.bind_execution_sliding(partial_promote_run) ||
      !fetch_bound(partial_promote_authority, *malformed_plan,
                   partial_promote_run, 0u, partial_promote_work) ||
      !partial_promote_sliding.issue_execution_sliding_promote(
          *malformed_plan, partial_promote_run, partial_promote_work.projection,
          partial_promote_work.uses, partial_promote) ||
      partial_promote.expected_bytes() != 16u ||
      !partial_promote_sliding.terminal_execution_sliding_promote(
          partial_promote_run, partial_promote,
          Status::fail(Reason::BackendFailed), execution::TerminalKind::Known,
          false, 8u) ||
      !partial_promote_sliding.release_execution_sliding_promote(
          partial_promote_run, std::move(partial_promote)) ||
      !partial_promote_run.prepare_final(partial_promote_frozen) ||
      !partial_promote_sliding.prepare_execution_sliding(
          *malformed_plan, partial_promote_frozen, partial_promote_prepared) ||
      !partial_promote_sliding.accept_execution_sliding(
          std::move(partial_promote_prepared), partial_promote_receipt) ||
      !partial_promote_run.accept_final(partial_promote_frozen,
                                        std::move(partial_promote_receipt),
                                        evidence) ||
      !evidence.has_failure || !evidence.first_failure_may_write ||
      evidence.status.reason() != Reason::BackendFailed ||
      evidence.promote_calls != 1u || evidence.promote_bytes != 8u) {
    return 65;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::sliding_authority
