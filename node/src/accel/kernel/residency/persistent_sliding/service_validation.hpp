#pragma once

#include "identity.hpp"
#include "validation.hpp"

#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

// Kept as a raw accessor for older private backend call sites. Zero means
// that no physical prefix has been accepted; it is never widened to Q.
[[nodiscard]] inline std::uint64_t persistent_sliding_accepted_end(
    const PersistentResidencySlidingControl &control) noexcept {
  return control.accepted_end;
}

// The backend may discover a physical terminal fact while the run is still
// associated with Control. Fold that fact while the Control gate is held so
// every later status, counter, and Final decision reads one authority.
inline void persistent_sliding_mark_unknown_locked(
    PersistentResidencySlidingControl &control,
    const rund::AccelCheck failure =
        rund::AccelCheck{false, "compute_device_lost"}) noexcept {
  control.quarantined = true;
  if (control.first_failure.ok) {
    control.first_failure = failure;
  }
}

[[nodiscard]] inline bool persistent_sliding_activate_control_locked(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request,
    PersistentResidencySlidingControl &control) noexcept {
  if (!persistent_sliding_request_valid(capability, request) ||
      (request.mode == PersistentResidencySlidingMode::BackendChunked &&
       !persistent_sliding_chunk_valid(capability, request))) {
    return false;
  }
  if (control.active || control.native != nullptr) {
    return false;
  }
  control.native = request.lowering;
  control.capability = capability;
  control.roles = request.roles;
  control.plan_identity = request.plan_identity;
  control.token = request.token;
  control.generation = request.generation;
  control.owner_nonce = request.owner_nonce;
  control.coordinate_count = request.coordinate_count;
  control.mode = request.mode;
  control.chunk_first = request.first_coordinate;
  control.chunk_span =
      request.mode == PersistentResidencySlidingMode::OneSubmit
          ? request.coordinate_count
          : request.chunk_count;
  control.accepted_end = 0u;
  control.chunk_submit_count = 0u;
  control.next_ready_coordinate = 0u;
  control.next_wait_coordinate = 0u;
  control.next_observation_coordinate = 0u;
  control.next_acknowledgement_coordinate = 0u;
  // Submission counters are committed by the common service only after the
  // backend reports acceptance.  Backend activation owns no accepted work.
  control.native_submit_count = 0u;
  control.queue_calls = 0u;
  control.epoch_native_submit_count = 0u;
  control.backend_epoch_callback_count = 0u;
  control.backing_wait_count = 0u;
  control.backing_signal_count = 0u;
  control.backing_acknowledgement_count = 0u;
  control.backing_service_failure_count = 0u;
  control.failed_admission_count = 0u;
  control.first_failed_admission_coordinate = PersistentSlidingNoCoordinate;
  control.first_service_failure_coordinate = PersistentSlidingNoCoordinate;
  control.gpu_completed_coordinates = 0u;
  control.completed_prefix = 0u;
  control.suppressed_first = PersistentSlidingNoCoordinate;
  control.suppressed_count = 0u;
  control.first_failure_coordinate = PersistentSlidingNoCoordinate;
  control.first_failure = {true, "ok"};
  control.final_callback_count = 0u;
  control.width = request.width;
  control.active = true;
  control.service_failed = false;
  control.quarantined = false;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_accept_chunk_locked(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request,
    const PersistentResidencySlidingControl &control) noexcept {
  return control.active && !control.quarantined &&
         request.mode == control.mode &&
         control.mode == PersistentResidencySlidingMode::BackendChunked &&
         persistent_sliding_chunk_valid(capability, request) &&
         request.first_coordinate == control.accepted_end &&
         request.first_coordinate == control.chunk_first &&
         request.chunk_count == control.chunk_span;
}

// This is the only admission predicate for a backend-reported physical
// submit.  It is deliberately const: accepting a submit and advancing the
// raw prefix are one later, gate-held commit.
[[nodiscard]] inline bool persistent_sliding_accept_valid(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request,
    const PersistentResidencySlidingControl &control,
    const std::uint64_t first, const std::uint64_t count) noexcept {
  if (!persistent_sliding_product_capable(capability) || !control.active ||
      control.quarantined ||
      request.mode != capability.mode || request.mode != control.mode ||
      count == 0u || first != control.accepted_end ||
      first >= request.coordinate_count ||
      count > request.coordinate_count - first ||
      control.chunk_first != first || control.chunk_span != count ||
      control.native_submit_count != control.queue_calls ||
      control.queue_calls != control.chunk_submit_count ||
      control.native_submit_count == std::numeric_limits<std::uint64_t>::max() ||
      control.queue_calls == std::numeric_limits<std::uint64_t>::max() ||
      control.chunk_submit_count == std::numeric_limits<std::uint64_t>::max() ||
      control.accepted_end >
          std::numeric_limits<std::uint64_t>::max() - count) {
    return false;
  }
  if (request.mode == PersistentResidencySlidingMode::OneSubmit) {
    return request.chunk_count == 0u && first == 0u &&
           control.accepted_end == 0u && count == request.coordinate_count;
  }
  return persistent_sliding_chunk_valid(capability, request);
}

// Commit the accepted prefix and all physical counters together.  Backends
// report acceptance only; they do not own this state transition.
[[nodiscard]] inline bool persistent_sliding_commit_accept_locked(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request,
    PersistentResidencySlidingControl &control, const std::uint64_t first,
    const std::uint64_t count) noexcept {
  if (!persistent_sliding_accept_valid(capability, request, control, first,
                                       count)) {
    return false;
  }
  control.accepted_end = first + count;
  ++control.native_submit_count;
  ++control.queue_calls;
  ++control.chunk_submit_count;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_activate_control(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request,
    PersistentResidencySlidingControl &control) noexcept {
  std::lock_guard lock{control.gate};
  return persistent_sliding_activate_control_locked(capability, request,
                                                    control);
}

[[nodiscard]] inline bool persistent_sliding_accept_ready_signal(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingReadySignal &signal) noexcept {
  std::lock_guard lock{control.gate};
  PersistentResidencySlidingServiceIdentity expected{};
  const bool failed_admission = !signal.admission.ok;
  if (!persistent_sliding_service_identity(control, signal.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, signal.identity) ||
      control.quarantined ||
      signal.identity.coordinate >= control.accepted_end ||
      signal.identity.coordinate != control.next_ready_coordinate ||
      (failed_admission && (signal.admission.reason == nullptr ||
                            signal.admission.reason[0] == '\0')) ||
      (control.failed_admission_count != 0u && !failed_admission) ||
      (control.service_failed && !failed_admission) ||
      (signal.identity.coordinate >= control.width &&
       signal.identity.coordinate - control.width >=
           control.next_acknowledgement_coordinate)) {
    return false;
  }
  if (failed_admission) {
    if (control.failed_admission_count == 0u) {
      control.first_failed_admission_coordinate = signal.identity.coordinate;
    }
    ++control.failed_admission_count;
  }
  ++control.next_ready_coordinate;
  ++control.backing_signal_count;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_accept_done_wait(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingDoneWait &wait) noexcept {
  std::lock_guard lock{control.gate};
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(control, wait.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, wait.identity) ||
      control.quarantined ||
      wait.identity.coordinate >= control.accepted_end ||
      wait.identity.coordinate != control.next_wait_coordinate ||
      wait.identity.coordinate >= control.next_ready_coordinate) {
    return false;
  }
  ++control.next_wait_coordinate;
  ++control.backing_wait_count;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_accept_done_observation(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingDoneWait &wait,
    const PersistentResidencySlidingDoneObservation &observation) noexcept {
  if (!persistent_sliding_same_service_identity(wait.identity,
                                                observation.identity) ||
      (observation.check.ok &&
       (observation.terminal != NativeTerminal::Known ||
        !observation.dispatched || !observation.completed ||
        !observation.may_write))) {
    return false;
  }
  std::lock_guard lock{control.gate};
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(control, wait.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, wait.identity) ||
      control.quarantined ||
      observation.identity.coordinate >= control.accepted_end ||
      observation.identity.coordinate != control.next_observation_coordinate ||
      observation.identity.coordinate >= control.next_wait_coordinate) {
    return false;
  }
  ++control.next_observation_coordinate;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_accept_done_acknowledgement(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingAcknowledgeDone &acknowledgement) noexcept {
  if (!acknowledgement.service_check.ok &&
      (acknowledgement.service_check.reason == nullptr ||
       acknowledgement.service_check.reason[0] == '\0')) {
    return false;
  }
  std::lock_guard lock{control.gate};
  PersistentResidencySlidingServiceIdentity expected{};
  const bool failure_open =
      control.service_failed || control.failed_admission_count != 0u;
  if (!persistent_sliding_service_identity(
          control, acknowledgement.identity.coordinate, expected) ||
      !persistent_sliding_same_service_identity(expected,
                                                acknowledgement.identity) ||
      control.quarantined ||
      acknowledgement.identity.coordinate >= control.accepted_end ||
      acknowledgement.identity.coordinate !=
          control.next_acknowledgement_coordinate ||
      acknowledgement.identity.coordinate >=
          control.next_observation_coordinate ||
      (!acknowledgement.service_check.ok && !failure_open)) {
    return false;
  }
  ++control.next_acknowledgement_coordinate;
  ++control.backing_acknowledgement_count;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_final_ready(
    const PersistentResidencySlidingControl &control) noexcept {
  std::lock_guard lock{control.gate};
  const std::uint64_t expected =
      control.mode == PersistentResidencySlidingMode::OneSubmit
          ? 1u
          : persistent_sliding_chunk_count(control.coordinate_count);
  return control.active && !control.service_failed && !control.quarantined &&
         control.native_submit_count == expected &&
         control.queue_calls == expected &&
         control.epoch_native_submit_count == 0u &&
         control.backend_epoch_callback_count == 0u &&
         control.failed_admission_count == 0u &&
         control.accepted_end == control.coordinate_count &&
         control.chunk_submit_count == expected &&
         control.next_acknowledgement_coordinate == control.coordinate_count &&
         control.backing_acknowledgement_count == control.coordinate_count;
}

[[nodiscard]] inline bool persistent_sliding_accept_service_failure(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingServiceFailure &failure) noexcept {
  if (failure.check.ok || failure.check.reason == nullptr ||
      failure.check.reason[0] == '\0') {
    return false;
  }
  std::lock_guard lock{control.gate};
  PersistentResidencySlidingServiceIdentity expected{};
  const bool next_input =
      failure.identity.coordinate == control.next_ready_coordinate;
  const bool completed_output =
      control.next_observation_coordinate != 0u &&
      failure.identity.coordinate + 1u == control.next_observation_coordinate;
  if (!persistent_sliding_service_identity(control, failure.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, failure.identity) ||
      control.service_failed ||
          failure.identity.coordinate >= control.accepted_end ||
      (!next_input && !completed_output)) {
    return false;
  }
  control.service_failed = true;
  control.quarantined = failure.terminal == NativeTerminal::UnknownMayWrite;
  control.first_service_failure_coordinate = failure.identity.coordinate;
  ++control.backing_service_failure_count;
  return true;
}

} // namespace rund::node::accel::detail
