#include "../internal.hpp"

#include "../../../../../accel/kernel/residency/persistent_sliding/service_validation.hpp"

#include <algorithm>

namespace rund::compute::detail::sliding_product_detail {

namespace {

[[nodiscard]] bool accepted_raw(
    const node::accel::detail::PersistentResidencySlidingControl &control) noexcept {
  return control.accepted_end != 0u || control.chunk_submit_count != 0u ||
         control.native_submit_count != 0u || control.queue_calls != 0u;
}

[[nodiscard]] node::accel::detail::PersistentResidencySlidingFinal
make_pending_final(SlidingProductRun &state) noexcept {
  const auto request = state.persistent_preparation.active_request();
  node::accel::detail::PersistentResidencySlidingFinal final{};
  final.check = {false, "compute_device_lost"};
  final.terminal = node::accel::detail::NativeTerminal::UnknownMayWrite;
  std::lock_guard lock{state.persistent_control.gate};
  final.evidence.plan_identity = request.plan_identity;
  final.evidence.token = request.token;
  final.evidence.generation = request.generation;
  final.evidence.owner_nonce = request.owner_nonce;
  final.evidence.cell_id = request.cell_id;
  final.evidence.cell_domain = request.cell_domain;
  final.evidence.coordinate_count = request.coordinate_count;
  final.evidence.accepted_coordinates = state.persistent_control.accepted_end;
  final.evidence.gpu_completed_coordinates =
      state.persistent_control.gpu_completed_coordinates;
  final.evidence.completed_prefix = state.persistent_control.completed_prefix;
  final.evidence.native_submit_count =
      state.persistent_control.native_submit_count;
  final.evidence.epoch_native_submit_count =
      state.persistent_control.epoch_native_submit_count;
  final.evidence.backend_epoch_callback_count =
      state.persistent_control.backend_epoch_callback_count;
  final.evidence.backing_wait_count = state.persistent_control.backing_wait_count;
  final.evidence.backing_signal_count =
      state.persistent_control.backing_signal_count;
  final.evidence.backing_acknowledgement_count =
      state.persistent_control.backing_acknowledgement_count;
  final.evidence.backing_service_failure_count =
      state.persistent_control.backing_service_failure_count;
  final.evidence.failed_admission_count =
      state.persistent_control.failed_admission_count;
  final.evidence.first_failed_admission_coordinate =
      state.persistent_control.first_failed_admission_coordinate;
  final.evidence.first_service_failure_coordinate =
      state.persistent_control.first_service_failure_coordinate;
  state.persistent_control.final_callback_count = 1u;
  final.evidence.suppressed_first =
      state.persistent_control.suppressed_first;
  final.evidence.suppressed_count = state.persistent_control.suppressed_count;
  final.evidence.queue_calls = state.persistent_control.queue_calls;
  final.evidence.accepted_end = state.persistent_control.accepted_end;
  final.evidence.chunk_submit_count =
      state.persistent_control.chunk_submit_count;
  final.evidence.first_failure_coordinate =
      state.persistent_control.first_failure_coordinate;
  final.evidence.width = request.width;
  return final;
}

void finish_pending(SlidingProductRun &state) noexcept {
  final_persistent(&state, make_pending_final(state));
}

} // namespace

Status submit_persistent(SlidingProductRun &state) noexcept {
  const auto &service = state.persistent_preparation.backend.service;
  if (!state.persistent_preparation || service.submit_result == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  auto request = state.persistent_preparation.active_request();
  request.lowering = state.persistent_preparation.backend.lowering;
  std::uint64_t first = 0u;
  std::uint64_t count = 0u;
  {
    std::lock_guard lock{state.persistent_control.gate};
    first = state.persistent_control.active
                ? state.persistent_control.accepted_end
                : 0u;
    if (first >= request.coordinate_count ||
        (state.persistent_control.active &&
         state.persistent_control.mode != request.mode)) {
      return Status::fail(Reason::CompletionInvalid);
    }
    const auto mode = state.persistent_control.active
                          ? state.persistent_control.mode
                          : request.mode;
    if (mode == node::accel::detail::PersistentResidencySlidingMode::OneSubmit) {
      if (request.chunk_count != 0u) {
        return Status::fail(Reason::CompletionInvalid);
      }
      count = request.coordinate_count - first;
    } else {
      if (request.chunk_count != 0u &&
          request.chunk_count >
              state.persistent_control.capability.max_chunk_span) {
        return Status::fail(Reason::CompletionInvalid);
      }
      count = std::min<std::uint64_t>(
          state.persistent_control.capability.max_chunk_span == 0u
              ? std::uint64_t{2u}
              : state.persistent_control.capability.max_chunk_span,
          request.coordinate_count - first);
    }
    if (count == 0u || count > request.coordinate_count - first) {
      return Status::fail(Reason::CompletionInvalid);
    }
    request.first_coordinate = first;
    request.chunk_count =
        mode == node::accel::detail::PersistentResidencySlidingMode::BackendChunked
            ? count
            : 0u;
    state.persistent_control.chunk_first = first;
    state.persistent_control.chunk_span = count;
  }

  node::accel::detail::PersistentResidencySlidingSubmitResult result =
      service.submit_result(request, state.persistent_control);

  bool accepted = false;
  switch (result.event) {
  case node::accel::detail::PersistentResidencySlidingSubmitEvent::Declined:
  case node::accel::detail::PersistentResidencySlidingSubmitEvent::Unknown:
    break;
  case node::accel::detail::PersistentResidencySlidingSubmitEvent::Accepted:
  case node::accel::detail::PersistentResidencySlidingSubmitEvent::AcceptedUnknown:
    {
      std::lock_guard lock{state.persistent_control.gate};
      accepted = node::accel::detail::persistent_sliding_commit_accept_locked(
          state.persistent_control.capability, request,
          state.persistent_control, first, count);
      if (!accepted) {
        result.status = {false, "compute_device_lost"};
        result.event = node::accel::detail::
            PersistentResidencySlidingSubmitEvent::AcceptedUnknown;
      }
    }
    break;
  }

  const bool unknown =
      result.event ==
          node::accel::detail::PersistentResidencySlidingSubmitEvent::Unknown ||
      result.event == node::accel::detail::
                          PersistentResidencySlidingSubmitEvent::AcceptedUnknown;
  if (unknown) {
    if (service.quarantine_unknown != nullptr) {
      service.quarantine_unknown(state.persistent_control);
    } else {
      std::lock_guard lock{state.persistent_control.gate};
      state.persistent_control.active = false;
      state.persistent_control.quarantined = true;
    }
    finish_pending(state);
    return result.status.ok ? Status::fail(Reason::DeviceLost)
                            : status_from(result.status);
  }

  if (!result.status.ok) {
    bool prior = false;
    {
      std::lock_guard lock{state.persistent_control.gate};
      prior = accepted_raw(state.persistent_control);
    }
    if (accepted || prior) {
      finish_pending(state);
    }
    return status_from(result.status);
  }
  if (!accepted) {
    return Status::fail(Reason::CompletionInvalid);
  }
  return Status::success();
}

bool persistent_submission_started(const SlidingProductRun &state) noexcept {
  std::lock_guard lock{state.persistent_control.gate};
  return accepted_raw(state.persistent_control);
}

} // namespace rund::compute::detail::sliding_product_detail
