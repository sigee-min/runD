#include "local.hpp"

#include <algorithm>
#include <concepts>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline_residency {
namespace {

namespace accel = rund::node::accel::detail;
namespace test = persistent_sliding_test;

template <typename Request>
concept HasProject = requires(Request request) { request.project; };

template <typename Request>
concept HasRelease = requires(Request request) { request.release; };

template <typename Request>
concept HasReturned = requires(Request request) { request.returned; };

static_assert(!HasProject<accel::PersistentResidencySlidingRequest>);
static_assert(!HasRelease<accel::PersistentResidencySlidingRequest>);
static_assert(!HasReturned<accel::PersistentResidencySlidingRequest>);

struct PersistentSlidingWait final {
  accel::PersistentResidencySlidingRequest request{};
  accel::PersistentResidencySlidingFinal observed{};
  std::uint64_t final_count{};
  bool valid{};
};

void CompletePersistentSliding(
    void *const raw, accel::PersistentResidencySlidingFinal &&final) noexcept {
  auto *const wait = static_cast<PersistentSlidingWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  ++wait->final_count;
  wait->valid = accel::persistent_sliding_final_valid(wait->request, final);
  wait->observed = std::move(final);
}

[[nodiscard]] accel::PersistentResidencySlidingCapability
Capability() noexcept {
  return accel::PersistentResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = accel::ResidencySlidingMemory::HostCoherent,
      .retained_bytes = sizeof(test::FakePersistentSlidingState),
      .width = 2u,
      .whole_run_preencoded = true,
      .device_generated_recurrence = false,
      .fixed_native_storage = false,
      .fixed_common_storage = false,
      .host_epoch_callbacks_zero = true,
      .mode = accel::PersistentResidencySlidingMode::OneSubmit,
  };
}

void BuildRequest(
    PersistentSlidingWait &wait,
    const std::shared_ptr<test::FakePersistentSlidingState> &state,
    const std::uint64_t coordinate_count) {
  wait = {};
  for (std::size_t slot = 0u; slot < state->capability.width; ++slot) {
    wait.request.roles[slot] = accel::PersistentResidencySlidingRole{
        .prepared = std::make_shared<std::uint8_t>(
            static_cast<std::uint8_t>(slot + 1u)),
        .locals = {0u},
        .local_count = 1u,
        .first_control_generation = static_cast<std::uint32_t>(slot + 1u),
        .control_generation_stride = state->capability.width,
        .first_descriptor_generation = slot + 11u,
        .descriptor_generation_stride = state->capability.width,
        .slot = static_cast<std::uint8_t>(slot),
    };
  }
  wait.request.plan_identity = 101u;
  wait.request.token = 202u;
  wait.request.generation = 303u;
  wait.request.owner_nonce = 404u;
  wait.request.cell_id = 505u;
  wait.request.cell_domain = 606u;
  wait.request.coordinate_count = coordinate_count;
  wait.request.tail_local_count = 1u;
  wait.request.lowering = state;
  wait.request.admission = std::make_shared<std::uint8_t>(1u);
  wait.request.final = CompletePersistentSliding;
  wait.request.user = &wait;
  wait.request.memory = state->capability.memory;
  wait.request.width = state->capability.width;
}

[[nodiscard]] bool SubmitAccepted(
    const accel::PersistentResidencySlidingRequest &request,
    accel::PersistentResidencySlidingControl &control) noexcept {
  const auto result = test::FakeServiceOps.submit_result(request, control);
  if (!result.status.ok ||
      result.event != accel::PersistentResidencySlidingSubmitEvent::Accepted) {
    return false;
  }
  std::lock_guard lock{control.gate};
  return accel::persistent_sliding_commit_accept_locked(
      control.capability, request, control, 0u, request.coordinate_count);
}

[[nodiscard]] bool CapabilityBoundaryCase() noexcept {
  accel::PersistentResidencySlidingCapability capability = Capability();
  if (!accel::persistent_sliding_product_capable(capability)) {
    return false;
  }
  if (accel::persistent_sliding_gpu_driven_capable(capability)) {
    return false;
  }
  capability.mode = accel::PersistentResidencySlidingMode::BackendChunked;
  if (!accel::persistent_sliding_product_capable(capability) ||
      accel::persistent_sliding_gpu_driven_capable(capability)) {
    return false;
  }
  capability.mode = static_cast<accel::PersistentResidencySlidingMode>(0xffu);
  if (accel::persistent_sliding_product_capable(capability)) {
    return false;
  }
  capability.mode = accel::PersistentResidencySlidingMode::OneSubmit;
  capability.whole_run_preencoded = false;
  if (accel::persistent_sliding_product_capable(capability)) {
    return false;
  }
  capability.whole_run_preencoded = true;
  capability.device_generated_recurrence = true;
  if (accel::persistent_sliding_gpu_driven_capable(capability)) {
    return false;
  }
  capability.fixed_native_storage = true;
  if (accel::persistent_sliding_gpu_driven_capable(capability)) {
    return false;
  }
  capability.fixed_common_storage = true;
  if (!accel::persistent_sliding_gpu_driven_capable(capability)) {
    return false;
  }
  capability.host_epoch_callbacks_zero = false;
  return !accel::persistent_sliding_product_capable(capability);
}

[[nodiscard]] bool TailBoundaryCase() noexcept {
  auto state = std::make_shared<test::FakePersistentSlidingState>();
  state->capability = Capability();
  PersistentSlidingWait wait{};
  BuildRequest(wait, state, 5u);
  if (!accel::persistent_sliding_request_valid(state->capability,
                                               wait.request)) {
    return false;
  }
  wait.request.tail_local_count = 0u;
  if (accel::persistent_sliding_request_valid(state->capability,
                                              wait.request)) {
    return false;
  }
  wait.request.tail_local_count = 2u;
  return !accel::persistent_sliding_request_valid(state->capability,
                                                  wait.request);
}

[[nodiscard]] bool ServiceIdentity(
    const accel::PersistentResidencySlidingRequest &request,
    const std::uint64_t coordinate,
    accel::PersistentResidencySlidingServiceIdentity &identity) noexcept {
  return accel::persistent_sliding_service_identity(request, coordinate,
                                                    identity);
}

[[nodiscard]] bool AdmissionBoundaryCase() noexcept {
  auto state = std::make_shared<test::FakePersistentSlidingState>();
  state->capability = Capability();
  PersistentSlidingWait wait{};
  BuildRequest(wait, state, 5u);
  accel::PersistentResidencySlidingControl control{};
  if (!SubmitAccepted(wait.request, control)) {
    return false;
  }
  accel::PersistentResidencySlidingServiceIdentity first{};
  accel::PersistentResidencySlidingServiceIdentity second{};
  if (!ServiceIdentity(wait.request, 0u, first) ||
      !ServiceIdentity(wait.request, 1u, second)) {
    return false;
  }
  if (test::FakeServiceOps
          .signal_ready(control,
                        accel::PersistentResidencySlidingReadySignal{
                            .identity = first,
                            .admission = {false, nullptr},
                        })
          .ok ||
      !test::FakeServiceOps
           .signal_ready(control,
                         accel::PersistentResidencySlidingReadySignal{
                             .identity = first,
                             .admission = {false, "backing_failed"},
                         })
           .ok ||
      test::FakeServiceOps
          .signal_ready(control,
                        accel::PersistentResidencySlidingReadySignal{
                            .identity = second,
                            .admission = {true, "ok"},
                        })
          .ok) {
    return false;
  }
  return test::FakeServiceOps
             .signal_ready(control,
                           accel::PersistentResidencySlidingReadySignal{
                               .identity = second,
                               .admission = {false, "backing_failed"},
                           })
             .ok &&
         control.failed_admission_count == 2u &&
         control.first_failed_admission_coordinate == 0u &&
         wait.final_count == 0u;
}

[[nodiscard]] bool
PersistentSlidingCase(const std::uint64_t coordinate_count) noexcept {
  auto state = std::make_shared<test::FakePersistentSlidingState>();
  state->capability = Capability();
  PersistentSlidingWait wait{};
  BuildRequest(wait, state, coordinate_count);
  accel::PersistentResidencySlidingControl control{};
  if (!SubmitAccepted(wait.request, control)) {
    return false;
  }

  accel::PersistentResidencySlidingServiceIdentity first{};
  accel::PersistentResidencySlidingServiceIdentity second{};
  if (!ServiceIdentity(wait.request, 0u, first) ||
      !ServiceIdentity(wait.request, 1u, second) ||
      test::FakeServiceOps
          .signal_ready(control,
                        accel::PersistentResidencySlidingReadySignal{
                            .identity = second,
                        })
          .ok) {
    return false;
  }
  accel::PersistentResidencySlidingServiceIdentity malformed = first;
  ++malformed.descriptor_generation;
  if (test::FakeServiceOps
          .signal_ready(control,
                        accel::PersistentResidencySlidingReadySignal{
                            .identity = malformed,
                        })
          .ok) {
    return false;
  }

  const std::uint64_t initial =
      std::min<std::uint64_t>(coordinate_count, wait.request.width);
  for (std::uint64_t coordinate = 0u; coordinate < initial; ++coordinate) {
    accel::PersistentResidencySlidingServiceIdentity identity{};
    if (!ServiceIdentity(wait.request, coordinate, identity) ||
        !test::FakeServiceOps
             .signal_ready(control,
                           accel::PersistentResidencySlidingReadySignal{
                               .identity = identity,
                           })
             .ok) {
      return false;
    }
  }

  if (test::FakeServiceOps
          .ack_done(control,
                    accel::PersistentResidencySlidingAcknowledgeDone{
                        .identity = first,
                        .service_check = {true, "ok"},
                    })
          .ok) {
    return false;
  }

  for (std::uint64_t coordinate = 0u; coordinate < coordinate_count;
       ++coordinate) {
    accel::PersistentResidencySlidingServiceIdentity identity{};
    accel::PersistentResidencySlidingDoneObservation observation{};
    if (!ServiceIdentity(wait.request, coordinate, identity) ||
        !test::FakeServiceOps
             .wait_done(control,
                        accel::PersistentResidencySlidingDoneWait{
                            .identity = identity,
                        },
                        observation)
             .ok ||
        !accel::persistent_sliding_same_service_identity(
            identity, observation.identity) ||
        wait.final_count != 0u ||
        !test::FakeServiceOps
             .ack_done(control,
                       accel::PersistentResidencySlidingAcknowledgeDone{
                           .identity = identity,
                           .service_check = {true, "ok"},
                       })
             .ok) {
      return false;
    }
    const std::uint64_t next = coordinate + wait.request.width;
    if (next < coordinate_count) {
      accel::PersistentResidencySlidingServiceIdentity next_identity{};
      if (!ServiceIdentity(wait.request, next, next_identity) ||
          !test::FakeServiceOps
               .signal_ready(control,
                             accel::PersistentResidencySlidingReadySignal{
                                 .identity = next_identity,
                             })
               .ok) {
        return false;
      }
    }
  }

  const accel::PersistentResidencySlidingEvidence &evidence =
      wait.observed.evidence;
  return wait.valid && state->submit_count == 1u &&
         state->epoch_native_submit_count == 0u &&
         state->backend_epoch_callback_count == 0u &&
         state->host_epoch_callback_count == 0u && wait.final_count == 1u &&
         state->gpu_completed_coordinates == coordinate_count &&
         evidence.native_submit_count == 1u &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_wait_count == coordinate_count &&
         evidence.backing_signal_count == coordinate_count &&
         evidence.backing_acknowledgement_count == coordinate_count &&
         evidence.final_callback_count == 1u && evidence.queue_calls == 1u &&
         evidence.gpu_completed_coordinates == coordinate_count &&
         !test::FakeServiceOps
              .ack_done(control,
                        accel::PersistentResidencySlidingAcknowledgeDone{
                            .identity = first,
                            .service_check = {true, "ok"},
                        })
              .ok &&
         wait.final_count == 1u;
}

} // namespace

int CheckPersistentSliding() {
  if (!CapabilityBoundaryCase()) {
    std::fprintf(stderr, "persistent sliding capability boundary failed\n");
    return 1;
  }
  if (!TailBoundaryCase()) {
    std::fprintf(stderr, "persistent sliding tail boundary failed\n");
    return 2;
  }
  if (!AdmissionBoundaryCase()) {
    std::fprintf(stderr, "persistent sliding admission boundary failed\n");
    return 3;
  }
  for (const std::uint64_t coordinate_count : {5u, 9u, 257u}) {
    if (!PersistentSlidingCase(coordinate_count)) {
      std::fprintf(stderr, "persistent sliding Q=%llu failed\n",
                   static_cast<unsigned long long>(coordinate_count));
      return 4;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
