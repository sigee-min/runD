#include "local.hpp"

#include <array>
#include <memory>
#include <thread>
#include <utility>

namespace rund_node_test_pipeline_residency {
namespace {
namespace execution = sliding_detail::execution;
namespace residency = sliding_detail::residency;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::Type;
using sliding_detail::admit;
using sliding_detail::admit_bound;
using sliding_detail::direct_plan;
using sliding_detail::drain;
using sliding_detail::drain_bound;
using sliding_detail::EpochWork;
using sliding_detail::fetch;
using sliding_detail::fetch_bound;
using sliding_detail::region;
using sliding_detail::register_direct;
} // namespace

int CheckSlidingTerminalUnknown() {
  execution::SlidingEvidence evidence{};
  // Unknown retains exact native/output cells as quarantine storage. Final
  // evidence wakes the owner, but quiescence cannot be fabricated.
  execution::Sliding unknown = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 107u, 10u, 1u,
      1u);
  EpochWork lost{};
  if (!unknown || !fetch(unknown, 0u, lost) || !admit(unknown, lost) ||
      !unknown.native_terminal(lost.native, Status::fail(Reason::DeviceLost),
                               execution::TerminalKind::UnknownMayWrite,
                               true) ||
      !unknown.close_model(evidence) || !evidence.quarantined ||
      evidence.terminal != execution::TerminalKind::UnknownMayWrite ||
      unknown.quiescent()) {
    return 19;
  }
  return 0;
}

int CheckSlidingTerminalFrontier() {
  execution::SlidingEvidence evidence{};
  // A future fetch failure freezes only its suffix. Ready prefix work remains
  // admissible, and a partial Host mutation needs explicit invalidation.
  execution::Sliding frontier = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(3u, 1u)), 110u, 16u, 2u,
      2u);
  EpochWork prefix0{};
  EpochWork prefix1{};
  EpochWork failed2{};
  execution::SlidingTicket failed_fetch{};
  execution::SlidingTicket rejected_fetch{};
  if (!frontier || !fetch(frontier, 0u, prefix0) ||
      !frontier.project(2u, failed2.uses, failed2.projection) ||
      !frontier.issue_fetch(failed2.projection, failed2.uses, 0u,
                            failed_fetch) ||
      !frontier.fetch_terminal(failed_fetch,
                               Status::fail(Reason::BackendFailed),
                               execution::TerminalKind::Known, true, 8u) ||
      frontier.close_model(evidence) || !frontier.invalidate(failed_fetch) ||
      !admit(frontier, prefix0) || !drain(frontier, prefix0, true) ||
      !fetch(frontier, 1u, prefix1) || !admit(frontier, prefix1) ||
      !drain(frontier, prefix1, true) ||
      frontier.issue_fetch(failed2.projection, failed2.uses, 0u,
                           rejected_fetch) ||
      !frontier.close_model(evidence) || !evidence.has_failure ||
      evidence.first_failure.ordinal != 2u ||
      !evidence.first_failure_may_write || evidence.admitted != 2u ||
      evidence.first_unsent != 2u) {
    return 22;
  }
  return 0;
}

int CheckSlidingTerminalFailures() {
  execution::SlidingEvidence evidence{};
  // Contradictory success+Unknown is consumed as Unknown quarantine, never
  // downgraded to a successful HostReady receipt.
  execution::Sliding malformed = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 111u, 17u, 1u,
      1u);
  EpochWork malformed_work{};
  execution::SlidingTicket malformed_fetch{};
  if (!malformed ||
      !malformed.project(0u, malformed_work.uses, malformed_work.projection) ||
      !malformed.issue_fetch(malformed_work.projection, malformed_work.uses, 0u,
                             malformed_fetch) ||
      !malformed.fetch_terminal(malformed_fetch, Status::success(),
                                execution::TerminalKind::UnknownMayWrite, true,
                                0u) ||
      !malformed.close_model(evidence) || !evidence.has_failure ||
      !evidence.quarantined ||
      evidence.terminal != execution::TerminalKind::UnknownMayWrite) {
    return 23;
  }

  // Known native may-write cannot retire/reuse the Device owner before the
  // exact mutation invalidation acknowledgement.
  execution::Sliding mutation = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 112u, 18u, 1u,
      1u);
  EpochWork mutation_work{};
  if (!mutation || !fetch(mutation, 0u, mutation_work) ||
      !admit(mutation, mutation_work) ||
      !mutation.native_terminal(mutation_work.native,
                                Status::fail(Reason::BackendFailed),
                                execution::TerminalKind::Known, true) ||
      mutation.close_model(evidence) ||
      !mutation.invalidate(mutation_work.native) ||
      !mutation.close_model(evidence) || !evidence.first_failure_may_write) {
    return 24;
  }

  auto graph_fixture = sliding_detail::graph_fixture();
  if (graph_fixture.owner == nullptr) {
    return 25;
  }
  std::shared_ptr<const residency::ResidencyPlan> graph_owner =
      std::move(graph_fixture.owner);
  const std::array<std::uint64_t, 3u> graph_bytes = graph_fixture.bytes;
  execution::SlidingInvocation graph_invocation =
      execution::SlidingInvocation::graph(graph_owner, 1u, graph_bytes);
  graph_owner.reset();

  // A failed Drain may cancel never-issued outputs, but it cannot erase a
  // sibling Drain callback that was already accepted.
  execution::Sliding drain_failure =
      execution::Sliding::create(graph_invocation, 113u, 19u, 1u, 2u);
  EpochWork drain_work{};
  execution::SlidingTicket drain_a{};
  execution::SlidingTicket drain_b{};
  if (!drain_failure || !fetch(drain_failure, 0u, drain_work) ||
      !admit(drain_failure, drain_work) ||
      !drain_failure.native_terminal(drain_work.native, Status::success(),
                                     execution::TerminalKind::Known, true) ||
      !drain_failure.issue_drain(drain_work.native, 1u, drain_a) ||
      !drain_failure.issue_drain(drain_work.native, 2u, drain_b) ||
      !drain_failure.drain_terminal(
          drain_a, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::Known, false, 0u) ||
      drain_failure.close_model(evidence) ||
      !drain_failure.drain_terminal(drain_b, Status::success(),
                                    execution::TerminalKind::Known, true,
                                    drain_b.expected_bytes) ||
      drain_failure.close_model(evidence) ||
      !drain_failure.invalidate(drain_b) ||
      !drain_failure.invalidate(drain_work.native) ||
      !drain_failure.close_model(evidence)) {
    return 25;
  }
  return 0;
}

int CheckSlidingTerminalSuffix() {
  execution::SlidingEvidence evidence{};
  // A later Promote may complete while an earlier Persist is pending. If that
  // Persist fails, the unadmitted Device mutation is explicitly invalidated;
  // it cannot become an immortal Ready cell or a dispatched suffix.
  execution::Sliding promoted_suffix = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(2u, 1u)), 115u, 21u, 2u,
      2u);
  EpochWork persisted0{};
  EpochWork promoted1{};
  execution::SlidingTicket drain0{};
  execution::SlidingTicket persist0{};
  execution::SlidingTicket promote1{};
  if (!promoted_suffix || !fetch(promoted_suffix, 0u, persisted0) ||
      !admit(promoted_suffix, persisted0) ||
      !promoted_suffix.native_terminal(persisted0.native, Status::success(),
                                       execution::TerminalKind::Known, true) ||
      !promoted_suffix.issue_drain(persisted0.native, 1u, drain0) ||
      !promoted_suffix.drain_terminal(drain0, Status::success(),
                                      execution::TerminalKind::Known, true,
                                      drain0.expected_bytes) ||
      !promoted_suffix.issue_persist(drain0, persist0) ||
      !fetch(promoted_suffix, 1u, promoted1) ||
      !promoted_suffix.issue_promote(promoted1.projection, promoted1.uses,
                                     promote1) ||
      !promoted_suffix.promote_terminal(promote1, Status::success(),
                                        execution::TerminalKind::Known, true,
                                        promote1.expected_bytes) ||
      !promoted_suffix.persist_terminal(
          persist0, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::Known, false, 0u) ||
      promoted_suffix.admit(promoted1.projection, promoted1.uses,
                            promoted1.native) ||
      promoted_suffix.close_model(evidence) ||
      !promoted_suffix.invalidate(promote1) ||
      !promoted_suffix.close_model(evidence) || evidence.admitted != 1u ||
      evidence.persist_frontier != 1u ||
      evidence.persist_completed_after_failure != 0u) {
    return 28;
  }

  execution::Sliding persist_mutation = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 116u, 22u, 1u,
      1u);
  EpochWork persist_work{};
  execution::SlidingTicket persist_drain{};
  execution::SlidingTicket persist_write{};
  if (!persist_mutation || !fetch(persist_mutation, 0u, persist_work) ||
      !admit(persist_mutation, persist_work) ||
      !persist_mutation.native_terminal(persist_work.native, Status::success(),
                                        execution::TerminalKind::Known, true) ||
      !persist_mutation.issue_drain(persist_work.native, 1u, persist_drain) ||
      !persist_mutation.drain_terminal(persist_drain, Status::success(),
                                       execution::TerminalKind::Known, true,
                                       persist_drain.expected_bytes) ||
      !persist_mutation.issue_persist(persist_drain, persist_write) ||
      !persist_mutation.persist_terminal(
          persist_write, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::Known, true, 8u) ||
      persist_mutation.close_model(evidence) ||
      !persist_mutation.invalidate(persist_write) ||
      !persist_mutation.close_model(evidence) ||
      evidence.persist_issued != 1u || evidence.persist_frontier != 1u) {
    return 29;
  }

  // Unknown quarantines its exact cell, but it is not permission to discard
  // another already-issued callback.  The first failure coordinate keeps its
  // own certainty while the run-wide terminal remains the worst certainty.
  execution::Sliding ordered_unknown = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(2u, 1u)), 117u, 23u, 2u,
      1u);
  EpochWork unknown0{};
  EpochWork unknown1{};
  execution::SlidingTicket unknown_fetch0{};
  execution::SlidingTicket unknown_fetch1{};
  if (!ordered_unknown ||
      !ordered_unknown.project(0u, unknown0.uses, unknown0.projection) ||
      !ordered_unknown.project(1u, unknown1.uses, unknown1.projection) ||
      !ordered_unknown.issue_fetch(unknown0.projection, unknown0.uses, 0u,
                                   unknown_fetch0) ||
      !ordered_unknown.issue_fetch(unknown1.projection, unknown1.uses, 0u,
                                   unknown_fetch1) ||
      !ordered_unknown.fetch_terminal(
          unknown_fetch1, Status::fail(Reason::DeviceLost),
          execution::TerminalKind::UnknownMayWrite, true, 0u) ||
      ordered_unknown.close_model(evidence) ||
      !ordered_unknown.fetch_terminal(
          unknown_fetch0, Status::fail(Reason::BackendFailed),
          execution::TerminalKind::Known, false, 0u) ||
      !ordered_unknown.close_model(evidence) || !evidence.quarantined ||
      evidence.first_failure.ordinal != 0u ||
      evidence.first_failure_terminal != execution::TerminalKind::Known ||
      evidence.terminal != execution::TerminalKind::UnknownMayWrite) {
    return 30;
  }

  // Per-coordinate capacity is a cold admission law, not retryable pressure.
  // Likewise, future Forecast cannot consume the slots reserved for the next
  // coordinate and strand the only progress-producing Fetch.
  if (execution::Sliding::create(
          execution::SlidingInvocation::direct(direct_plan(2u, 2u)), 118u, 24u,
          1u, 2u)) {
    return 31;
  }
  execution::Sliding reserved = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(4u, 1u)), 119u, 25u, 1u,
      1u);
  std::array<EpochWork, 4u> reserved_work{};
  execution::SlidingTicket future{};
  if (!reserved ||
      !reserved.project(3u, reserved_work[3u].uses,
                        reserved_work[3u].projection) ||
      reserved.issue_fetch(reserved_work[3u].projection, reserved_work[3u].uses,
                           0u, future)) {
    return 32;
  }
  for (std::size_t epoch = 0u; epoch < reserved_work.size(); ++epoch) {
    if (!fetch(reserved, epoch, reserved_work[epoch]) ||
        !admit(reserved, reserved_work[epoch]) ||
        !drain(reserved, reserved_work[epoch], true)) {
      return 32;
    }
  }
  if (!reserved.close_model(evidence)) {
    return 32;
  }

  // A success receipt with a wrong byte extent is internally contradictory.
  // Its may_write bit is not trusted; every write-capable phase requires the
  // exact invalidation acknowledgement before close/reuse.
  execution::Sliding bad_fetch = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 120u, 26u, 1u,
      1u);
  EpochWork bad_fetch_work{};
  execution::SlidingTicket bad_fetch_ticket{};
  if (!bad_fetch ||
      !bad_fetch.project(0u, bad_fetch_work.uses, bad_fetch_work.projection) ||
      !bad_fetch.issue_fetch(bad_fetch_work.projection, bad_fetch_work.uses, 0u,
                             bad_fetch_ticket) ||
      !bad_fetch.fetch_terminal(bad_fetch_ticket, Status::success(),
                                execution::TerminalKind::Known, false,
                                bad_fetch_ticket.expected_bytes - 1u) ||
      bad_fetch.close_model(evidence) ||
      !bad_fetch.invalidate(bad_fetch_ticket) ||
      !bad_fetch.close_model(evidence) || !evidence.first_failure_may_write ||
      evidence.fetch_calls != 1u) {
    return 33;
  }
  execution::Sliding bad_promote = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 121u, 27u, 1u,
      1u);
  EpochWork bad_promote_work{};
  execution::SlidingTicket bad_promote_ticket{};
  if (!bad_promote || !fetch(bad_promote, 0u, bad_promote_work) ||
      !bad_promote.issue_promote(bad_promote_work.projection,
                                 bad_promote_work.uses, bad_promote_ticket) ||
      !bad_promote.promote_terminal(bad_promote_ticket, Status::success(),
                                    execution::TerminalKind::Known, false,
                                    bad_promote_ticket.expected_bytes - 1u) ||
      bad_promote.close_model(evidence) ||
      !bad_promote.invalidate(bad_promote_ticket) ||
      !bad_promote.close_model(evidence) || evidence.promote_calls != 1u) {
    return 34;
  }
  execution::Sliding bad_drain = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 122u, 28u, 1u,
      1u);
  EpochWork bad_drain_work{};
  execution::SlidingTicket bad_drain_ticket{};
  if (!bad_drain || !fetch(bad_drain, 0u, bad_drain_work) ||
      !admit(bad_drain, bad_drain_work) ||
      !bad_drain.native_terminal(bad_drain_work.native, Status::success(),
                                 execution::TerminalKind::Known, true) ||
      !bad_drain.issue_drain(bad_drain_work.native, 1u, bad_drain_ticket) ||
      !bad_drain.drain_terminal(bad_drain_ticket, Status::success(),
                                execution::TerminalKind::Known, false,
                                bad_drain_ticket.expected_bytes - 1u) ||
      bad_drain.close_model(evidence) ||
      !bad_drain.invalidate(bad_drain_ticket) ||
      !bad_drain.invalidate(bad_drain_work.native) ||
      !bad_drain.close_model(evidence) || evidence.drain_calls != 1u) {
    return 35;
  }
  execution::Sliding bad_persist = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(1u, 1u)), 123u, 29u, 1u,
      1u);
  EpochWork bad_persist_work{};
  execution::SlidingTicket bad_persist_drain{};
  execution::SlidingTicket bad_persist_ticket{};
  if (!bad_persist || !fetch(bad_persist, 0u, bad_persist_work) ||
      !admit(bad_persist, bad_persist_work) ||
      !bad_persist.native_terminal(bad_persist_work.native, Status::success(),
                                   execution::TerminalKind::Known, true) ||
      !bad_persist.issue_drain(bad_persist_work.native, 1u,
                               bad_persist_drain) ||
      !bad_persist.drain_terminal(bad_persist_drain, Status::success(),
                                  execution::TerminalKind::Known, true,
                                  bad_persist_drain.expected_bytes) ||
      !bad_persist.issue_persist(bad_persist_drain, bad_persist_ticket) ||
      !bad_persist.persist_terminal(bad_persist_ticket, Status::success(),
                                    execution::TerminalKind::Known, false,
                                    bad_persist_ticket.expected_bytes - 1u) ||
      bad_persist.close_model(evidence) ||
      !bad_persist.invalidate(bad_persist_ticket) ||
      !bad_persist.close_model(evidence) || evidence.persist_calls != 1u ||
      evidence.persist_frontier != 1u) {
    return 36;
  }

  auto graph_fixture = sliding_detail::graph_fixture();
  if (graph_fixture.owner == nullptr) {
    return 37;
  }
  std::shared_ptr<const residency::ResidencyPlan> graph_owner =
      std::move(graph_fixture.owner);
  const std::array<std::uint64_t, 3u> graph_bytes = graph_fixture.bytes;
  execution::SlidingInvocation graph_invocation =
      execution::SlidingInvocation::graph(graph_owner, 1u, graph_bytes);
  graph_owner.reset();

  // Unknown Drain retains already-issued siblings until their callbacks, but
  // cancels never-issued Reserved siblings which were never exposed to Host
  // mutation. Neither path can wedge final quiescence.
  execution::Sliding unknown_sibling =
      execution::Sliding::create(graph_invocation, 125u, 31u, 1u, 2u);
  EpochWork unknown_sibling_work{};
  execution::SlidingTicket unknown_drain0{};
  execution::SlidingTicket unknown_drain1{};
  if (!unknown_sibling || !fetch(unknown_sibling, 0u, unknown_sibling_work) ||
      !admit(unknown_sibling, unknown_sibling_work) ||
      !unknown_sibling.native_terminal(unknown_sibling_work.native,
                                       Status::success(),
                                       execution::TerminalKind::Known, true) ||
      !unknown_sibling.issue_drain(unknown_sibling_work.native, 1u,
                                   unknown_drain0) ||
      !unknown_sibling.issue_drain(unknown_sibling_work.native, 2u,
                                   unknown_drain1) ||
      !unknown_sibling.drain_terminal(
          unknown_drain0, Status::fail(Reason::DeviceLost),
          execution::TerminalKind::UnknownMayWrite, true, 0u) ||
      unknown_sibling.close_model(evidence) ||
      !unknown_sibling.drain_terminal(unknown_drain1, Status::success(),
                                      execution::TerminalKind::Known, true,
                                      unknown_drain1.expected_bytes) ||
      !unknown_sibling.close_model(evidence) || !evidence.quarantined) {
    return 37;
  }
  execution::Sliding unknown_unissued =
      execution::Sliding::create(graph_invocation, 126u, 32u, 1u, 2u);
  EpochWork unknown_unissued_work{};
  execution::SlidingTicket unknown_only{};
  if (!unknown_unissued ||
      !fetch(unknown_unissued, 0u, unknown_unissued_work) ||
      !admit(unknown_unissued, unknown_unissued_work) ||
      !unknown_unissued.native_terminal(unknown_unissued_work.native,
                                        Status::success(),
                                        execution::TerminalKind::Known, true) ||
      !unknown_unissued.issue_drain(unknown_unissued_work.native, 1u,
                                    unknown_only) ||
      !unknown_unissued.drain_terminal(
          unknown_only, Status::fail(Reason::DeviceLost),
          execution::TerminalKind::UnknownMayWrite, true, 0u) ||
      !unknown_unissued.close_model(evidence) || !evidence.quarantined) {
    return 38;
  }
  return 0;
}
} // namespace rund_node_test_pipeline_residency
