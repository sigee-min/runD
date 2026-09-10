#include "local.hpp"

#include <utility>

namespace rund_node_test_pipeline_residency::service_free_direct_test::
    authority_detail {

[[nodiscard]] bool SuccessCase(const std::uint64_t iterations) noexcept {
  Fixture fixture{};
  if (!fixture.initialize(iterations)) {
    return false;
  }
  const residency::DirectRecurrenceRequest request = fixture.request();
  residency::DirectRecurrenceLease lease =
      fixture.authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal final{};
  Publication publication{};
  const auto snapshot = fixture.registration->snapshot();
  const std::array regions{snapshot.bindings[0u].region,
                           snapshot.bindings[1u].region};
  if (!lease || fixture.authority().release_frames(regions) ||
      fixture.registration->release() != residency::RegistrationResult::Busy ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          lease, rund::compute::Status::success(),
          execution::TerminalKind::Known, true, iterations, final) ||
      !Finish(fixture, lease, std::move(final), publication) ||
      publication.count != 1u || !publication.success || final ||
      !fixture.release()) {
    return false;
  }
  residency::DirectRecurrenceFinal replay{};
  return !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
      lease, rund::compute::Status::success(), execution::TerminalKind::Known,
      true, iterations, replay);
}

[[nodiscard]] bool KnownRetryCase() noexcept {
  Fixture fixture{};
  if (!fixture.initialize()) {
    return false;
  }
  const residency::DirectRecurrenceRequest request = fixture.request();
  residency::DirectRecurrenceLease first =
      fixture.authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal failed{};
  Publication failure_publication{};
  if (!first ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          first,
          rund::compute::Status::fail(rund::compute::Reason::BackendFailed),
          execution::TerminalKind::Known, false, 0u, failed) ||
      !Finish(fixture, first, std::move(failed), failure_publication) ||
      failure_publication.count != 1u || failure_publication.success) {
    return false;
  }

  Fixture retry_fixture{};
  if (!retry_fixture.initialize()) {
    return false;
  }
  residency::DirectRecurrenceLease retry =
      retry_fixture.authority().direct_recurrences().begin_direct_recurrence(
          retry_fixture.request());
  residency::DirectRecurrenceFinal success{};
  Publication success_publication{};
  return retry &&
         retry_fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
             retry, rund::compute::Status::success(),
             execution::TerminalKind::Known, true, 5u, success) &&
         Finish(retry_fixture, retry, std::move(success),
                success_publication) &&
         success_publication.count == 1u && success_publication.success &&
         retry_fixture.release();
}

[[nodiscard]] bool KnownPartialRetryCase() noexcept {
  Fixture fixture{};
  if (!fixture.initialize()) {
    return false;
  }
  const residency::DirectRecurrenceRequest request = fixture.request();
  residency::DirectRecurrenceLease first =
      fixture.authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal failed{};
  Publication failure_publication{};
  if (!first ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          first,
          rund::compute::Status::fail(rund::compute::Reason::BackendFailed),
          execution::TerminalKind::Known, true, 2u, failed) ||
      !Finish(fixture, first, std::move(failed), failure_publication) ||
      failure_publication.count != 1u || failure_publication.success) {
    return false;
  }

  Fixture retry_fixture{};
  if (!retry_fixture.initialize()) {
    return false;
  }
  residency::DirectRecurrenceLease retry =
      retry_fixture.authority().direct_recurrences().begin_direct_recurrence(
          retry_fixture.request());
  residency::DirectRecurrenceFinal success{};
  Publication success_publication{};
  return retry &&
         retry_fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
             retry, rund::compute::Status::success(),
             execution::TerminalKind::Known, true, 5u, success) &&
         Finish(retry_fixture, retry, std::move(success),
                success_publication) &&
         success_publication.count == 1u && success_publication.success &&
         retry_fixture.release();
}

[[nodiscard]] bool KnownNoWritePartialRetryCase() noexcept {
  Fixture fixture{};
  if (!fixture.initialize()) {
    return false;
  }
  const residency::DirectRecurrenceRequest request = fixture.request();
  residency::DirectRecurrenceLease first =
      fixture.authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal failed{};
  Publication failure_publication{};
  if (!first ||
      !fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
          first,
          rund::compute::Status::fail(rund::compute::Reason::ScanSumOverflow),
          execution::TerminalKind::Known, false, 2u, failed) ||
      !Finish(fixture, first, std::move(failed), failure_publication) ||
      failure_publication.count != 1u || failure_publication.success) {
    return false;
  }

  Fixture retry_fixture{};
  if (!retry_fixture.initialize()) {
    return false;
  }
  residency::DirectRecurrenceLease retry =
      retry_fixture.authority().direct_recurrences().begin_direct_recurrence(
          retry_fixture.request());
  residency::DirectRecurrenceFinal success{};
  Publication success_publication{};
  return retry &&
         retry_fixture.authority().direct_recurrences().prepare_direct_recurrence_final(
             retry, rund::compute::Status::success(),
             execution::TerminalKind::Known, true, 5u, success) &&
         Finish(retry_fixture, retry, std::move(success),
                success_publication) &&
         success_publication.count == 1u && success_publication.success &&
         retry_fixture.release();
}

} // namespace
  // rund_node_test_pipeline_residency::service_free_direct_test::authority_detail
