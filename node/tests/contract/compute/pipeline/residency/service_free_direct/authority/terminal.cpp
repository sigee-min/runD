#include "local.hpp"

#include <array>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency::service_free_direct_test::
    authority_detail {

[[nodiscard]] bool FrozenBusyCase() noexcept {
  auto idle_registry = std::make_shared<residency::Registry>();
  residency::DirectRecurrenceLease idle_lease{};
  residency::DirectRecurrenceFinal idle_final{};
  if (idle_registry->authority().direct_recurrences().abort_direct_recurrence(
          idle_lease,
          rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid),
          execution::TerminalKind::Known,
          false) != residency::DirectAbort::Invalid ||
      idle_registry->authority().direct_recurrences().abort_direct_recurrence_frozen(
          std::move(idle_final)) != residency::DirectAbort::Invalid) {
    return false;
  }

  Fixture fixture{};
  Fixture foreign{};
  if (!fixture.initialize() || !foreign.initialize()) {
    return false;
  }
  residency::DirectRecurrenceLease lease =
      fixture.authority().direct_recurrences().begin_direct_recurrence(fixture.request());
  residency::DirectRecurrenceLease foreign_lease =
      foreign.authority().direct_recurrences().begin_direct_recurrence(foreign.request());
  residency::DirectRecurrenceFinal final{};
  residency::DirectRecurrenceFinal foreign_final{};
  if (!lease || !foreign_lease ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          lease, rund::compute::Status::success(),
          execution::TerminalKind::Known, true, 5u, final) ||
      !foreign.authority().direct_recurrences().prepare_direct_recurrence_final(
          foreign_lease, rund::compute::Status::success(),
          execution::TerminalKind::Known, true, 5u, foreign_final)) {
    return false;
  }

  const auto frozen_before = fixture.registration->snapshot();
  const auto token = lease.token();
  const auto generation = lease.generation();
  const auto owner = lease.owner();
  const auto foreign_final_abort =
      fixture.authority().direct_recurrences().abort_direct_recurrence_frozen(
          std::move(foreign_final));
  const auto stale_abort = fixture.authority().direct_recurrences().abort_direct_recurrence(
      lease,
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid),
      execution::TerminalKind::Known, false);
  const auto frozen_after = fixture.registration->snapshot();
  if (foreign_final_abort != residency::DirectAbort::Busy ||
      stale_abort != residency::DirectAbort::Busy ||
      frozen_before.state->phase() !=
          residency::registration_detail::Lifecycle::Frozen ||
      !SameSnapshot(frozen_before, frozen_after) ||
      !RowsBusy(fixture, frozen_before) || lease.token() != token ||
      lease.generation() != generation || lease.owner() != owner) {
    return false;
  }

  const residency::ExecutionClose staged =
      fixture.authority().direct_recurrences().stage_direct_recurrence_final(std::move(final));
  if (!staged || staged.quarantined) {
    return false;
  }
  const auto pending_before = fixture.registration->snapshot();
  const auto pending_abort = fixture.authority().direct_recurrences().abort_direct_recurrence(
      lease,
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid),
      execution::TerminalKind::Known, false);
  const auto pending_after = fixture.registration->snapshot();
  if (pending_abort != residency::DirectAbort::Busy ||
      pending_before.state->phase() !=
          residency::registration_detail::Lifecycle::Pending ||
      !SameSnapshot(pending_before, pending_after) ||
      !RowsBusy(fixture, pending_before) ||
      fixture.registration->release_pending(lease) !=
          residency::RegistrationResult::Done) {
    return false;
  }

  Publication publication{};
  publication.fixture = &fixture;
  publication.authority = &fixture.authority();
  publication.lease = &lease;
  publication.reenter = true;
  const residency::ExecutionClose closed =
      fixture.registration->finish_pending(lease, &publication, Publish);
  const bool foreign_preserved = static_cast<bool>(foreign_final);
  const auto foreign_closed =
      foreign.authority().direct_recurrences().abort_direct_recurrence_frozen(
          std::move(foreign_final));
  const bool foreign_released = foreign.release();
  return closed && closed.registration == residency::RegistrationResult::Done &&
         publication.count == 1u && publication.success &&
         publication.phase_before ==
             residency::registration_detail::Lifecycle::Inflight &&
         publication.phase_after ==
             residency::registration_detail::Lifecycle::Inflight &&
         publication.same_rows && publication.rows_before &&
         publication.rows_after && publication.credential_same &&
         publication.abort_result == residency::DirectAbort::Busy &&
         foreign_preserved &&
         foreign_closed == residency::DirectAbort::Closed && foreign_released &&
         fixture.release();
}

[[nodiscard]] bool UnknownAndForgeryCase() noexcept {
  Fixture fixture{};
  if (!fixture.initialize()) {
    return false;
  }
  Fixture foreign{};
  if (!foreign.initialize()) {
    return false;
  }
  residency::DirectRecurrenceRequest malformed = fixture.request();
  malformed.proof_owner = foreign.request().proof_owner;
  const auto before = fixture.registration->snapshot();
  if (fixture.authority().direct_recurrences().begin_direct_recurrence(malformed) ||
      !SameSnapshot(before, fixture.registration->snapshot()) ||
      before.state->phase() !=
          residency::registration_detail::Lifecycle::Active) {
    return false;
  }

  malformed = fixture.request();
  malformed.proof_owner = std::shared_ptr<const void>(
      fixture.registration.get(), [](const void *) noexcept {});
  if (fixture.authority().direct_recurrences().begin_direct_recurrence(malformed) ||
      !SameSnapshot(before, fixture.registration->snapshot())) {
    return false;
  }

  const residency::DirectRecurrenceRequest request = fixture.request();
  residency::DirectRecurrenceLease lease =
      fixture.authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal foreign_final{};
  residency::DirectRecurrenceFinal unknown{};
  Publication publication{};
  if (!lease ||
      foreign.authority().direct_recurrences().prepare_direct_recurrence_final(
          lease, rund::compute::Status::fail(rund::compute::Reason::DeviceLost),
          execution::TerminalKind::UnknownMayWrite, true, 0u, foreign_final) ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          lease, rund::compute::Status::fail(rund::compute::Reason::DeviceLost),
          execution::TerminalKind::UnknownMayWrite, true, 0u, unknown)) {
    return false;
  }
  {
    residency::DirectRecurrenceLease released = std::move(lease);
  }
  return fixture.authority()
             .direct_recurrences().stage_direct_recurrence_final(std::move(unknown))
             .quarantined &&
         publication.count == 0u &&
         !fixture.authority().direct_recurrences().begin_direct_recurrence(request);
}

} // namespace
  // rund_node_test_pipeline_residency::service_free_direct_test::authority_detail
