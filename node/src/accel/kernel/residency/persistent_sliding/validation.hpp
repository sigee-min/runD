#pragma once

#include "range.hpp"
#include "request.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline std::uint64_t persistent_sliding_chunk_count(
    const std::uint64_t coordinates,
    const std::uint64_t span = 2u) noexcept {
  if (coordinates == 0u || span == 0u) {
    return 0u;
  }
  return coordinates / span + static_cast<std::uint64_t>(coordinates % span != 0u);
}

[[nodiscard]] inline bool persistent_sliding_product_capable(
    const PersistentResidencySlidingCapability &capability) noexcept {
  const bool mode_ok =
      capability.mode == PersistentResidencySlidingMode::OneSubmit ||
      capability.mode == PersistentResidencySlidingMode::BackendChunked;
  return capability.check.ok && capability.width != 0u &&
         capability.width <= PersistentResidencySlidingCapacity &&
         capability.whole_run_preencoded && mode_ok &&
         capability.host_epoch_callbacks_zero &&
         (capability.mode == PersistentResidencySlidingMode::OneSubmit ||
          (capability.max_chunk_span != 0u &&
           capability.max_chunk_span <= 2u));
}

[[nodiscard]] inline bool persistent_sliding_chunk_valid(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request) noexcept {
  if (!persistent_sliding_product_capable(capability) ||
      capability.mode != PersistentResidencySlidingMode::BackendChunked ||
      capability.max_chunk_span == 0u || request.coordinate_count == 0u ||
      request.chunk_count == 0u ||
      request.chunk_count > capability.max_chunk_span ||
      request.chunk_count > request.coordinate_count ||
      request.first_coordinate >= request.coordinate_count ||
      request.first_coordinate >
          request.coordinate_count - request.chunk_count) {
    return false;
  }
  return true;
}

[[nodiscard]] inline bool persistent_sliding_gpu_driven_capable(
    const PersistentResidencySlidingCapability &capability) noexcept {
  return persistent_sliding_product_capable(capability) &&
         capability.device_generated_recurrence &&
         capability.fixed_native_storage && capability.fixed_common_storage;
}

[[nodiscard]] inline bool persistent_sliding_request_valid(
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &request) noexcept {
  if (!persistent_sliding_product_capable(capability) ||
      request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.owner_nonce == 0u ||
      request.cell_id == 0u || request.cell_domain == 0u ||
      request.coordinate_count == 0u || request.tail_local_count == 0u ||
      request.width != capability.width ||
      request.memory != capability.memory || request.lowering == nullptr ||
      request.admission == nullptr || request.final == nullptr ||
      request.user == nullptr) {
    return false;
  }
  if (request.mode != capability.mode) {
    return false;
  }
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    const PersistentResidencySlidingRole &role = request.roles[slot];
    if (role.prepared == nullptr || role.slot != slot ||
        role.local_count == 0u ||
        role.local_count > ResidencyWindowLocalCapacity ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        role.first_descriptor_generation == 0u ||
        role.descriptor_generation_stride == 0u) {
      return false;
    }
  }
  if (!persistent_sliding_range_valid(
          std::span<const PersistentResidencySlidingRole>{request.roles.data(),
                                                           request.width},
          request.width, request.coordinate_count)) {
    return false;
  }
  const std::size_t tail_slot =
      static_cast<std::size_t>((request.coordinate_count - 1u) % request.width);
  if (request.mode == PersistentResidencySlidingMode::OneSubmit &&
      (request.chunk_count != 0u || request.first_coordinate != 0u)) {
    return false;
  }
  if (request.mode == PersistentResidencySlidingMode::BackendChunked &&
      !persistent_sliding_chunk_valid(capability, request)) {
    return false;
  }
  return request.tail_local_count <= request.roles[tail_slot].local_count;
}

[[nodiscard]] inline std::uint64_t persistent_sliding_expected_chunks(
    const PersistentResidencySlidingRequest &request) noexcept {
  return request.mode == PersistentResidencySlidingMode::OneSubmit
             ? 1u
             : persistent_sliding_chunk_count(request.coordinate_count);
}

[[nodiscard]] inline bool persistent_sliding_final_counts_valid(
    const PersistentResidencySlidingRequest &request,
    const PersistentResidencySlidingEvidence &evidence) noexcept {
  const std::uint64_t expected =
      persistent_sliding_expected_chunks(request);
  return evidence.native_submit_count == expected &&
         evidence.queue_calls == expected &&
         evidence.chunk_submit_count ==
             expected &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.final_callback_count == 1u;
}

[[nodiscard]] inline bool persistent_sliding_counts_valid(
    const PersistentResidencySlidingRequest &request,
    const PersistentResidencySlidingEvidence &evidence) noexcept {
  const std::uint64_t expected =
      persistent_sliding_expected_chunks(request);
  return evidence.native_submit_count == evidence.queue_calls &&
         evidence.queue_calls == evidence.chunk_submit_count &&
         evidence.chunk_submit_count <= expected &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.final_callback_count == 1u;
}

[[nodiscard]] inline bool persistent_sliding_final_valid(
    const PersistentResidencySlidingRequest &request,
    const PersistentResidencySlidingFinal &final) noexcept {
  const PersistentResidencySlidingEvidence &evidence = final.evidence;
  if (evidence.plan_identity != request.plan_identity ||
      evidence.token != request.token ||
      evidence.generation != request.generation ||
      evidence.owner_nonce != request.owner_nonce ||
      evidence.cell_id != request.cell_id ||
      evidence.cell_domain != request.cell_domain ||
      evidence.coordinate_count != request.coordinate_count ||
      evidence.width != request.width ||
      evidence.accepted_coordinates > evidence.coordinate_count ||
      evidence.accepted_end > evidence.coordinate_count ||
      evidence.gpu_completed_coordinates > evidence.accepted_coordinates ||
      evidence.completed_prefix > evidence.gpu_completed_coordinates ||
      evidence.suppressed_count > evidence.accepted_coordinates ||
      evidence.backing_wait_count > evidence.coordinate_count ||
      evidence.backing_signal_count > evidence.coordinate_count ||
      evidence.backing_acknowledgement_count > evidence.coordinate_count ||
      evidence.backing_service_failure_count > 1u ||
      evidence.failed_admission_count > evidence.coordinate_count ||
      evidence.accepted_coordinates != evidence.accepted_end ||
      !persistent_sliding_counts_valid(request, evidence)) {
    return false;
  }
  if (evidence.suppressed_count == 0u) {
    if (evidence.suppressed_first != PersistentSlidingNoCoordinate) {
      return false;
    }
  } else if (evidence.suppressed_first >= evidence.accepted_coordinates ||
             evidence.suppressed_count >
                 evidence.accepted_coordinates - evidence.suppressed_first) {
    return false;
  }
  if (evidence.failed_admission_count == 0u) {
    if (evidence.first_failed_admission_coordinate !=
        PersistentSlidingNoCoordinate) {
      return false;
    }
  } else if (evidence.first_failed_admission_coordinate >=
                 evidence.coordinate_count ||
             evidence.failed_admission_count >
                 evidence.coordinate_count -
                     evidence.first_failed_admission_coordinate) {
    return false;
  }
  if ((evidence.backing_service_failure_count == 0u) !=
      (evidence.first_service_failure_coordinate ==
       PersistentSlidingNoCoordinate)) {
    return false;
  }
  if (evidence.backing_service_failure_count != 0u &&
      evidence.first_service_failure_coordinate >= evidence.coordinate_count) {
    return false;
  }
  if (final.check.ok) {
    return final.terminal == NativeTerminal::Known &&
           persistent_sliding_final_counts_valid(request, evidence) &&
           evidence.accepted_coordinates == evidence.coordinate_count &&
           evidence.accepted_end == evidence.coordinate_count &&
           evidence.gpu_completed_coordinates == evidence.coordinate_count &&
           evidence.completed_prefix == evidence.coordinate_count &&
           evidence.backing_wait_count == evidence.coordinate_count &&
           evidence.backing_signal_count == evidence.coordinate_count &&
           evidence.backing_acknowledgement_count ==
               evidence.coordinate_count &&
           evidence.backing_service_failure_count == 0u &&
           evidence.first_service_failure_coordinate ==
               PersistentSlidingNoCoordinate &&
           evidence.failed_admission_count == 0u &&
           evidence.first_failed_admission_coordinate ==
               PersistentSlidingNoCoordinate &&
           evidence.suppressed_count == 0u &&
           evidence.first_failure_coordinate == PersistentSlidingNoCoordinate;
  }
  if (final.terminal == NativeTerminal::UnknownMayWrite) {
    return evidence.first_failure_coordinate == PersistentSlidingNoCoordinate ||
           evidence.first_failure_coordinate < evidence.coordinate_count;
  }
  return final.terminal == NativeTerminal::Known &&
         evidence.first_failure_coordinate < evidence.coordinate_count;
}

} // namespace rund::node::accel::detail
