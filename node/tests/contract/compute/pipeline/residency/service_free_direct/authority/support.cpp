#include "local.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <array>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency::service_free_direct_test::
    authority_detail {

[[nodiscard]] bool SameSnapshot(
    const residency::DirectRecurrenceRegistration::Snapshot &left,
    const residency::DirectRecurrenceRegistration::Snapshot &right) noexcept {
  if (left.proof != right.proof || left.state != right.state ||
      left.count != right.count) {
    return false;
  }
  for (std::size_t index = 0u; index < left.count; ++index) {
    if (left.bindings[index].view != right.bindings[index].view ||
        left.bindings[index].region != right.bindings[index].region ||
        left.bindings[index].registration !=
            right.bindings[index].registration) {
      return false;
    }
  }
  return true;
}

struct ProofOwner final {
  rund::kernel::LoweringArtifact artifact{};
  std::array<rund::kernel::ComputeDispatchWindow, 1u> windows{};
};

[[nodiscard]] std::shared_ptr<accel::ServiceFreeDirectProof>
Proof(const std::uint64_t iterations) {
  auto proof = std::make_shared<accel::ServiceFreeDirectProof>();
  auto owner = std::make_shared<ProofOwner>();
  proof->identity = accel::ServiceFreeDirectIdentity{
      .hi = 0x5346524545444952ull, .lo = iterations};
  proof->semantic_owner = owner;
  owner->artifact.key.api = rund::kernel::ComputeApi::Metal;
  owner->artifact.key.variant =
      rund::kernel::LoweringArtifactVariant::Recurrence;
  owner->artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  owner->artifact.ok = true;
  owner->artifact.reason = "ok";
  proof->artifact = &owner->artifact;
  proof->plan.api = rund::kernel::ComputeApi::Metal;
  proof->plan.tile_count = 16u;
  proof->plan.input_buffer_count = 1u;
  proof->plan.output_buffer_count = 1u;
  proof->plan.dispatch_window_tiles = 16u;
  proof->plan.dispatch_count = 1u;
  proof->plan.ok = true;
  proof->plan.reason = "ok";
  proof->inputs[0u] = rund::kernel::ResidentBufferRef{
      .id = 11u,
      .bytes = 64u,
      .offset_bytes = 0u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 16u,
      .usage = rund::kernel::kResidentUsageRead};
  proof->input_handles[0u] = std::make_shared<std::uint8_t>(2u);
  proof->outputs[0u] = rund::kernel::ResidentBufferRef{
      .id = 22u,
      .bytes = 64u,
      .offset_bytes = 0u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 16u,
      .usage = rund::kernel::kResidentUsageWrite};
  proof->output_handles[0u] = std::make_shared<std::uint8_t>(3u);
  proof->states[0u] = rund::kernel::ResidentBufferRef{
      .id = 11u,
      .bytes = 64u,
      .offset_bytes = 0u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = 64u,
      .usage = rund::kernel::kResidentUsageRead};
  proof->state_handles[0u] = proof->input_handles[0u];
  proof->states[1u] = rund::kernel::ResidentBufferRef{
      .id = 22u,
      .bytes = 64u,
      .offset_bytes = 0u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = 64u,
      .usage = rund::kernel::kResidentUsageWrite};
  proof->state_handles[1u] = proof->output_handles[0u];
  owner->windows[0u] = rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                           .tile_count = 16u};
  proof->windows = owner->windows.data();
  proof->iterations = iterations;
  proof->window_count = owner->windows.size();
  proof->input_count = 1u;
  proof->output_count = 1u;
  proof->state_count = 2u;
  proof->fixed_common_storage = true;
  proof->retention = accel::ServiceFreeDirectRetention::Terminal;
  return proof;
}

[[nodiscard]] bool
Fixture::initialize(const std::uint64_t iterations) noexcept {
  registry = std::make_shared<residency::Registry>();
  registration =
      residency::register_direct_recurrence_proof(registry, Proof(iterations));
  return registration != nullptr;
}

[[nodiscard]] residency::Authority &Fixture::authority() noexcept {
  return registry->authority();
}

[[nodiscard]] residency::DirectRecurrenceRequest
Fixture::request() const noexcept {
  return registration == nullptr ? residency::DirectRecurrenceRequest{}
                                 : registration->request();
}

[[nodiscard]] bool Fixture::release() noexcept {
  return registration != nullptr &&
         registration->release() == residency::RegistrationResult::Done;
}

[[nodiscard]] bool
RowsBusy(Fixture &fixture,
         const residency::DirectRecurrenceRegistration::Snapshot
             &snapshot) noexcept {
  const std::array regions{snapshot.bindings[0u].region,
                           snapshot.bindings[1u].region};
  return !fixture.authority().release_frames(regions);
}

void Publish(void *const raw, const bool success) noexcept {
  auto *const publication = static_cast<Publication *>(raw);
  if (publication == nullptr) {
    return;
  }
  ++publication->count;
  publication->success = success;
  if (publication->reenter && publication->fixture != nullptr &&
      publication->authority != nullptr && publication->lease != nullptr) {
    const auto before = publication->fixture->registration->snapshot();
    publication->phase_before = before.state->phase();
    publication->rows_before = RowsBusy(*publication->fixture, before);
    const auto token = publication->lease->token();
    const auto generation = publication->lease->generation();
    const auto owner = publication->lease->owner();
    publication->abort_result =
        publication->authority->direct_recurrences().abort_direct_recurrence(
            *publication->lease,
            rund::compute::Status::fail(
                rund::compute::Reason::CompletionInvalid),
            execution::TerminalKind::Known, false);
    const auto after = publication->fixture->registration->snapshot();
    publication->phase_after = after.state->phase();
    publication->same_rows = SameSnapshot(before, after);
    publication->rows_after = RowsBusy(*publication->fixture, after);
    publication->credential_same =
        publication->lease->token() == token &&
        publication->lease->generation() == generation &&
        publication->lease->owner() == owner;
  }
}

[[nodiscard]] bool Finish(Fixture &fixture,
                          residency::DirectRecurrenceLease &lease,
                          residency::DirectRecurrenceFinal &&final,
                          Publication &publication) noexcept {
  const residency::ExecutionClose staged =
      fixture.authority().direct_recurrences().stage_direct_recurrence_final(
          std::move(final));
  if (!staged || staged.quarantined ||
      fixture.registration->release_pending(lease) !=
          residency::RegistrationResult::Done) {
    return false;
  }
  const residency::ExecutionClose closed =
      fixture.registration->finish_pending(lease, &publication, Publish);
  return closed && closed.registration == residency::RegistrationResult::Done;
}

} // namespace
  // rund_node_test_pipeline_residency::service_free_direct_test::authority_detail
