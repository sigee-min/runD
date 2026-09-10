#include "internal.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {
namespace {

struct Publication final {
  std::uint64_t count{};
  bool success{};
};

void Publish(void *const raw, const bool success) noexcept {
  auto *const publication = static_cast<Publication *>(raw);
  if (publication == nullptr) {
    return;
  }
  ++publication->count;
  publication->success = success;
}

} // namespace

[[nodiscard]] wait_detail::Result
CloseAuthority(wait_detail::Owner &owner,
               const accel::DeviceVsmFinal &final) noexcept {
  if (final.terminal == accel::DeviceVsmTerminal::UnknownMayWrite) {
    return wait_detail::dispose_unknown(owner, final.evidence.completed_epochs);
  }
  residency::DirectRecurrenceFinal prepared{};
  Publication publication{};
  if (owner.registry == nullptr || owner.registration == nullptr ||
      !owner.lease ||
      !owner.registry->authority().direct_recurrences().prepare_direct_recurrence_final(
          owner.lease,
          final.check.ok
              ? rund::compute::Status::success()
              : rund::compute::Status::fail(rund::compute::Reason::DeviceLost),
          residency::execution::TerminalKind::Known, final.evidence.may_write,
          final.evidence.completed_epochs, prepared) ||
      !owner.registry->authority().direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared)) ||
      owner.registration->release_pending(owner.lease) !=
          residency::RegistrationResult::Done ||
      !owner.registration->finish_pending(owner.lease, &publication, Publish) ||
      publication.count != 1u || publication.success != final.check.ok ||
      owner.registration->release() != residency::RegistrationResult::Done) {
    return wait_detail::Result::Failed;
  }
  return wait_detail::Result::Closed;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
