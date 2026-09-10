#include "local.hpp"

#include <memory>

namespace rund_node_test_pipeline_residency::persistent_sliding_test {
namespace {

[[nodiscard]] std::shared_ptr<FakePersistentSlidingState>
state_of(const accel::PersistentResidencySlidingControl &control) noexcept {
  return std::static_pointer_cast<FakePersistentSlidingState>(control.native);
}

void CompleteFakePersistentSliding(
    FakePersistentSlidingState &state,
    accel::PersistentResidencySlidingControl &control) noexcept {
  if (state.final_sent || !accel::persistent_sliding_final_ready(control)) {
    return;
  }
  state.final_sent = true;
  ++state.final_count;
  state.request.final(
      state.request.user,
      accel::PersistentResidencySlidingFinal{
          .check = {true, "ok"},
          .terminal = accel::NativeTerminal::Known,
          .evidence =
              accel::PersistentResidencySlidingEvidence{
                  .plan_identity = state.request.plan_identity,
                  .token = state.request.token,
                  .generation = state.request.generation,
                  .owner_nonce = state.request.owner_nonce,
                  .cell_id = state.request.cell_id,
                  .cell_domain = state.request.cell_domain,
                  .coordinate_count = state.request.coordinate_count,
                  .accepted_coordinates = control.accepted_end,
                  .gpu_completed_coordinates = state.gpu_completed_coordinates,
                  .completed_prefix = control.next_acknowledgement_coordinate,
                  .native_submit_count = control.native_submit_count,
                  .epoch_native_submit_count = state.epoch_native_submit_count,
                  .backend_epoch_callback_count =
                      state.backend_epoch_callback_count,
                  .host_epoch_callback_count = state.host_epoch_callback_count,
                  .backing_wait_count = control.backing_wait_count,
                  .backing_signal_count = control.backing_signal_count,
                  .backing_acknowledgement_count =
                      control.backing_acknowledgement_count,
                  .backing_service_failure_count =
                      control.backing_service_failure_count,
                  .failed_admission_count = control.failed_admission_count,
                  .first_failed_admission_coordinate =
                      control.first_failed_admission_coordinate,
                  .first_service_failure_coordinate =
                      control.first_service_failure_coordinate,
                  .final_callback_count = state.final_count,
                  .queue_calls = control.queue_calls,
                  .accepted_end = control.accepted_end,
                  .chunk_submit_count = control.chunk_submit_count,
                  .completed_ns = state.request.coordinate_count,
                  .width = state.request.width,
              },
      });
}

} // namespace

accel::PersistentResidencySlidingSubmitResult FakePersistentSlidingSubmit(
    const accel::PersistentResidencySlidingRequest &request,
    accel::PersistentResidencySlidingControl &control) noexcept {
  const auto state =
      std::static_pointer_cast<FakePersistentSlidingState>(request.lowering);
  if (state == nullptr || state->submit_count != 0u ||
      !accel::persistent_sliding_activate_control(state->capability, request,
                                                  control)) {
    return {{false, "accel_kernel_pipeline_invalid"},
            accel::PersistentResidencySlidingSubmitEvent::Declined};
  }
  state->request = request;
  state->request.lowering.reset();
  ++state->submit_count;
  return {{true, "ok"},
          accel::PersistentResidencySlidingSubmitEvent::Accepted};
}

rund::AccelCheck FakePersistentSlidingSignalReady(
    accel::PersistentResidencySlidingControl &control,
    const accel::PersistentResidencySlidingReadySignal &signal) noexcept {
  const auto state = state_of(control);
  if (state == nullptr ||
      !accel::persistent_sliding_accept_ready_signal(control, signal)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  return {true, "ok"};
}

rund::AccelCheck FakePersistentSlidingWaitDone(
    accel::PersistentResidencySlidingControl &control,
    const accel::PersistentResidencySlidingDoneWait &wait,
    accel::PersistentResidencySlidingDoneObservation &observation) noexcept {
  observation = {};
  const auto state = state_of(control);
  if (state == nullptr ||
      !accel::persistent_sliding_accept_done_wait(control, wait)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  observation = accel::PersistentResidencySlidingDoneObservation{
      .identity = wait.identity,
      .check = {true, "ok"},
      .terminal = accel::NativeTerminal::Known,
      .dispatched = true,
      .completed = true,
      .may_write = true,
  };
  if (!accel::persistent_sliding_accept_done_observation(control, wait,
                                                         observation)) {
    observation = {};
    return {false, "accel_kernel_pipeline_invalid"};
  }
  ++state->gpu_completed_coordinates;
  return {true, "ok"};
}

rund::AccelCheck FakePersistentSlidingAckDone(
    accel::PersistentResidencySlidingControl &control,
    const accel::PersistentResidencySlidingAcknowledgeDone
        &acknowledgement) noexcept {
  const auto state = state_of(control);
  if (state == nullptr ||
      !accel::persistent_sliding_accept_done_acknowledgement(control,
                                                             acknowledgement)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  CompleteFakePersistentSliding(*state, control);
  return {true, "ok"};
}

rund::AccelCheck FakePersistentSlidingFailService(
    accel::PersistentResidencySlidingControl &control,
    const accel::PersistentResidencySlidingServiceFailure &failure) noexcept {
  const auto state = state_of(control);
  return state != nullptr && accel::persistent_sliding_accept_service_failure(
                                 control, failure)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
}

} // namespace rund_node_test_pipeline_residency::persistent_sliding_test
