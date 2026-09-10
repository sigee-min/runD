#include "local.hpp"

#include "src/compute/device/residency/execution/direct_recurrence/registration.hpp"

#include <concepts>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency {
namespace {

namespace accel = rund::node::accel::detail;
namespace test = service_free_direct_test;

template <typename Request>
concept HasSignalReady = requires(Request request) { request.signal_ready; };

template <typename Request>
concept HasWaitDone = requires(Request request) { request.wait_done; };

template <typename Request>
concept HasAckDone = requires(Request request) { request.ack_done; };

template <typename Request>
concept HasFailService = requires(Request request) { request.fail_service; };

template <typename Request>
concept HasProject = requires(Request request) { request.project; };

template <typename Request>
concept HasRelease = requires(Request request) { request.release; };

template <typename Request>
concept HasReturned = requires(Request request) { request.returned; };

template <typename Request>
concept HasPageCount = requires(Request request) { request.page_count; };

static_assert(!HasSignalReady<accel::ServiceFreeDirectRequest>);
static_assert(!HasWaitDone<accel::ServiceFreeDirectRequest>);
static_assert(!HasAckDone<accel::ServiceFreeDirectRequest>);
static_assert(!HasFailService<accel::ServiceFreeDirectRequest>);
static_assert(!HasProject<accel::ServiceFreeDirectRequest>);
static_assert(!HasRelease<accel::ServiceFreeDirectRequest>);
static_assert(!HasReturned<accel::ServiceFreeDirectRequest>);
static_assert(!HasPageCount<accel::ServiceFreeDirectRequest>);

struct Wait final {
  accel::ServiceFreeDirectRequest request{};
  accel::ServiceFreeDirectFinal final{};
  std::uint64_t final_count{};
  bool valid{};
};

struct ProofOwner final {
  rund::kernel::LoweringArtifact artifact{};
  std::array<rund::kernel::ComputeDispatchWindow, 1u> windows{};
};

struct RegistrationPublication final {
  std::uint64_t count{};
  bool success{};
};

void PublishRegistration(void *const raw, const bool success) noexcept {
  auto *const publication = static_cast<RegistrationPublication *>(raw);
  if (publication == nullptr) {
    return;
  }
  ++publication->count;
  publication->success = success;
}

void Complete(void *const raw, accel::ServiceFreeDirectFinal &&final) noexcept {
  auto *const wait = static_cast<Wait *>(raw);
  if (wait == nullptr) {
    return;
  }
  ++wait->final_count;
  wait->valid = accel::service_free_direct_final_valid(wait->request, final);
  wait->final = std::move(final);
}

[[nodiscard]] accel::ServiceFreeDirectCapability Capability() noexcept {
  return accel::ServiceFreeDirectCapability{
      .check = {true, "ok"},
      .retained_bytes = sizeof(test::FakeState),
      .transient_bytes = 0u,
      .device_generated_recurrence = true,
      .fixed_native_storage = true,
      .fixed_common_storage = true,
      .one_native_submit = true,
      .host_service_turns_zero = true,
      .host_epoch_callbacks_zero = true,
      .aggregate_terminal_once = true,
      .terminal_output = true,
      .history_output = false,
  };
}

[[nodiscard]] std::shared_ptr<accel::ServiceFreeDirectProof>
Proof(const std::uint64_t iterations) {
  auto proof = std::make_shared<accel::ServiceFreeDirectProof>();
  auto owner = std::make_shared<ProofOwner>();
  proof->identity = accel::ServiceFreeDirectIdentity{
      .hi = 0x5346524545444952ull,
      .lo = iterations,
  };
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
      .usage = rund::kernel::kResidentUsageRead,
  };
  proof->input_handles[0u] = std::make_shared<std::uint8_t>(2u);
  proof->outputs[0u] = rund::kernel::ResidentBufferRef{
      .id = 22u,
      .bytes = 64u,
      .offset_bytes = 0u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 16u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  proof->output_handles[0u] = std::make_shared<std::uint8_t>(3u);
  proof->states[0u] = rund::kernel::ResidentBufferRef{
      .id = proof->inputs[0u].id,
      .bytes = proof->inputs[0u].bytes,
      .offset_bytes = 0u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = proof->inputs[0u].bytes,
      .usage = rund::kernel::kResidentUsageRead,
  };
  proof->state_handles[0u] = proof->input_handles[0u];
  proof->states[1u] = rund::kernel::ResidentBufferRef{
      .id = proof->outputs[0u].id,
      .bytes = proof->outputs[0u].bytes,
      .offset_bytes = 0u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = proof->outputs[0u].bytes,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  proof->state_handles[1u] = proof->output_handles[0u];
  owner->windows[0u] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = 16u,
  };
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

void BuildRequest(Wait &wait, const std::shared_ptr<test::FakeState> &state,
                  const std::uint64_t iterations) {
  wait = {};
  wait.request = accel::ServiceFreeDirectRequest{
      .proof = Proof(iterations),
      .lowering = state,
      .admission = std::make_shared<const std::uint8_t>(4u),
      .token = 101u,
      .generation = 202u,
      .nonce = 303u,
      .final = Complete,
      .user = &wait,
  };
}

[[nodiscard]] bool CapabilityBoundaryCase() noexcept {
  auto capability = Capability();
  if (!accel::service_free_direct_capable(capability)) {
    return false;
  }
  capability.fixed_common_storage = false;
  if (accel::service_free_direct_capable(capability)) {
    return false;
  }
  capability.fixed_common_storage = true;
  capability.host_service_turns_zero = false;
  return !accel::service_free_direct_capable(capability);
}

[[nodiscard]] bool AggregateCase(const std::uint64_t iterations) {
  auto state = std::make_shared<test::FakeState>();
  state->capability = Capability();
  Wait wait{};
  BuildRequest(wait, state, iterations);
  accel::ServiceFreeDirectPreparation preparation{
      .capability = state->capability,
      .lowering = state,
      .submit = test::FakeSubmit,
  };
  if (!preparation ||
      !accel::service_free_direct_request_valid(preparation.capability,
                                                wait.request) ||
      !preparation.submit(wait.request).ok || !wait.valid ||
      wait.final_count != 1u || state->submit_count != 1u ||
      state->final_count != 1u ||
      wait.final.evidence.iterations != iterations ||
      wait.final.evidence.completed_iterations != iterations ||
      wait.final.evidence.native_submit_count != 1u ||
      wait.final.evidence.epoch_native_submit_count != 0u ||
      wait.final.evidence.payload_dispatch_count != 1u ||
      wait.final.evidence.host_service_turn_count != 0u ||
      wait.final.evidence.host_epoch_callback_count != 0u ||
      wait.final.evidence.final_callback_count != 1u) {
    return false;
  }
  if (preparation.submit(wait.request).ok) {
    return false;
  }
  auto malformed = wait.final;
  ++malformed.evidence.epoch_native_submit_count;
  return !accel::service_free_direct_final_valid(wait.request, malformed);
}

[[nodiscard]] bool ProofBoundaryCase() {
  auto proof = Proof(5u);
  if (!accel::service_free_direct_proof_valid(*proof)) {
    return false;
  }
  proof->iterations = 1u;
  if (accel::service_free_direct_proof_valid(*proof)) {
    return false;
  }
  proof = Proof(5u);
  proof->input_handles[0u].reset();
  return !accel::service_free_direct_proof_valid(*proof);
}

[[nodiscard]] bool RegistrationJoinCase() {
  namespace residency = rund::compute::detail::residency;
  auto registry = std::make_shared<residency::Registry>();
  const auto proof = Proof(5u);
  const std::shared_ptr<residency::DirectRecurrenceRegistration> registration =
      residency::register_direct_recurrence_proof(registry, proof);
  const auto snapshot =
      registration == nullptr
          ? residency::DirectRecurrenceRegistration::Snapshot{}
          : registration->snapshot();
  if (registration == nullptr || snapshot.proof != proof ||
      snapshot.count != proof->state_count) {
    return false;
  }
  const residency::DirectRecurrenceRequest request = registration->request();
  residency::DirectRecurrenceLease lease =
      registry->authority().direct_recurrences().begin_direct_recurrence(request);
  residency::DirectRecurrenceFinal final{};
  RegistrationPublication publication{};
  return lease &&
         registration->release() == residency::RegistrationResult::Busy &&
         registry->authority().direct_recurrences().prepare_direct_recurrence_final(
             lease, rund::compute::Status::success(),
             residency::execution::TerminalKind::Known, true, 5u, final) &&
         registry->authority().direct_recurrences().stage_direct_recurrence_final(
             std::move(final)) &&
         registration->release_pending(lease) ==
             residency::RegistrationResult::Done &&
         registration->finish_pending(lease, &publication,
                                      PublishRegistration) &&
         publication.count == 1u && publication.success &&
         registration->release() == residency::RegistrationResult::Done;
}

} // namespace

int CheckServiceFreeDirect() {
  if (const int authority = test::CheckAuthority(); authority != 0) {
    return 10 + authority;
  }
  if (!CapabilityBoundaryCase()) {
    return 1;
  }
  if (!ProofBoundaryCase()) {
    return 2;
  }
  if (!RegistrationJoinCase()) {
    return 3;
  }
  constexpr std::uint64_t Iterations[]{5u, 9u, 257u};
  for (const std::uint64_t iterations : Iterations) {
    if (!AggregateCase(iterations)) {
      std::fprintf(stderr, "service-free direct aggregate q=%llu failed\n",
                   static_cast<unsigned long long>(iterations));
      return 4;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
