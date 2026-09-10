#include "actual.hpp"

#include <utility>

namespace rund_node_test_pipeline_residency::service_free_direct_test {
namespace {

namespace residency = rund::compute::detail::residency;

void Publish(void *const raw, const bool success) noexcept {
  auto *const run = static_cast<ActualAuthorityRun *>(raw);
  if (run == nullptr) {
    return;
  }
  ++run->publication_count;
  run->publication_success = success;
}

} // namespace

ActualAuthorityRun begin_actual_authority_run(
    std::shared_ptr<residency::Registry> registry,
    std::shared_ptr<const rund::node::accel::detail::ServiceFreeDirectProof>
        proof) noexcept {
  std::shared_ptr<residency::DirectRecurrenceRegistration> registration =
      residency::register_direct_recurrence_proof(registry, std::move(proof));
  if (registration == nullptr) {
    return {};
  }
  residency::DirectRecurrenceLease lease =
      registry->authority().direct_recurrences().begin_direct_recurrence(registration->request());
  return ActualAuthorityRun{std::move(registry), std::move(registration),
                            std::move(lease)};
}

bool finish_actual_authority_run(ActualAuthorityRun &run,
                                 const rund::compute::Status status) noexcept {
  if (!run) {
    return false;
  }
  residency::DirectRecurrenceFinal prepared{};
  const bool success = static_cast<bool>(status);
  if (!run.registry->authority().direct_recurrences().prepare_direct_recurrence_final(
          run.lease, status, residency::execution::TerminalKind::Known, true,
          success ? run.lease.iterations() : 0u, prepared) ||
      !run.registry->authority().direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared)) ||
      run.registration->release_pending(run.lease) !=
          residency::RegistrationResult::Done ||
      !run.registration->finish_pending(run.lease, &run, Publish) ||
      run.publication_count != 1u || run.publication_success != success ||
      run.registration->release() != residency::RegistrationResult::Done) {
    return false;
  }
  return true;
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
