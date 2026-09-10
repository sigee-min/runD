#include "local.hpp"

#include "src/compute/device/residency/execution/device_vsm/registration.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test {
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

bool CheckAuthority() {
  namespace residency = rund::compute::detail::residency;
  auto registry = std::make_shared<residency::Registry>();
  const auto proof = Proof(9u);
  const std::shared_ptr<residency::DeviceVsmRegistration> registration =
      residency::register_device_vsm_proof(registry, proof);
  const auto snapshot = registration == nullptr
                            ? residency::DeviceVsmRegistration::Snapshot{}
                            : registration->snapshot();
  const auto bindings = snapshot.binding_span();
  if (registration == nullptr || snapshot.proof != proof ||
      snapshot.count != proof->residents.count ||
      bindings[0u].region.role != residency::FrameRole::Input ||
      bindings[1u].region.role != residency::FrameRole::Output ||
      bindings[0u].view.resource != proof->residents.rows[0u].backing.id ||
      bindings[1u].view.resource != proof->residents.rows[1u].backing.id ||
      registration->request().iterations != proof->geometry.page_count) {
    return false;
  }
  residency::DirectRecurrenceLease lease =
      registry->authority().direct_recurrences().begin_direct_recurrence(registration->request());
  residency::DirectRecurrenceFinal final{};
  Publication publication{};
  if (!lease ||
      registration->release() != residency::RegistrationResult::Busy ||
      !registry->authority().direct_recurrences().prepare_direct_recurrence_final(
          lease, rund::compute::Status::success(),
          residency::execution::TerminalKind::Known, true,
          proof->geometry.page_count, final)) {
    return false;
  }
  const auto staged =
      registry->authority().direct_recurrences().stage_direct_recurrence_final(std::move(final));
  return staged &&
         registration->release_pending(lease) ==
             residency::RegistrationResult::Done &&
         registration->finish_pending(lease, &publication, Publish) &&
         publication.count == 1u && publication.success &&
         registration->release() == residency::RegistrationResult::Done;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
