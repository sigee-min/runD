#include "internal.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct FinalSnapshot final {
  PersistentResidencySlidingRequest request{};
  std::uint64_t submit_begin_ns{};
  NativeTerminal physical_terminal{NativeTerminal::Known};
};

[[nodiscard]] PersistentResidencySlidingEvidence
evidence(const FinalSnapshot &snapshot,
         const PersistentResidencySlidingControl &control, const bool success,
         const bool unknown) noexcept {
  const std::uint64_t failure =
      success ? PersistentSlidingNoCoordinate
              : (control.first_service_failure_coordinate !=
                         PersistentSlidingNoCoordinate
                     ? control.first_service_failure_coordinate
                     : (control.first_failed_admission_coordinate !=
                                PersistentSlidingNoCoordinate
                            ? control.first_failed_admission_coordinate
                            : (unknown ? PersistentSlidingNoCoordinate
                                       : (control.next_observation_coordinate == 0u
                                              ? PersistentSlidingNoCoordinate
                                              : control.next_observation_coordinate - 1u))));
    return PersistentResidencySlidingEvidence{
      .plan_identity = snapshot.request.plan_identity,
      .token = snapshot.request.token,
      .generation = snapshot.request.generation,
      .owner_nonce = snapshot.request.owner_nonce,
      .cell_id = snapshot.request.cell_id,
      .cell_domain = snapshot.request.cell_domain,
      .coordinate_count = snapshot.request.coordinate_count,
      .accepted_coordinates = control.accepted_end,
      .gpu_completed_coordinates = control.gpu_completed_coordinates,
      .completed_prefix = success ? snapshot.request.coordinate_count
                                  : (unknown ? control.gpu_completed_coordinates
                                             : failure),
      .suppressed_first = control.suppressed_first,
      .suppressed_count = control.suppressed_count,
      .native_submit_count = control.native_submit_count,
      .epoch_native_submit_count = control.epoch_native_submit_count,
      .backend_epoch_callback_count = control.backend_epoch_callback_count,
      .host_epoch_callback_count = 0u,
      .backing_wait_count = control.backing_wait_count,
      .backing_signal_count = control.backing_signal_count,
      .backing_acknowledgement_count = control.backing_acknowledgement_count,
      .backing_service_failure_count = control.backing_service_failure_count,
      .failed_admission_count = control.failed_admission_count,
      .first_failed_admission_coordinate =
          control.first_failed_admission_coordinate,
      .first_service_failure_coordinate =
          control.first_service_failure_coordinate,
      .final_callback_count = control.final_callback_count,
      .queue_calls = control.queue_calls,
      .accepted_end = control.accepted_end,
      .chunk_submit_count = control.chunk_submit_count,
      .first_failure_coordinate = failure,
      .completed_ns = MonotonicNanoseconds() - snapshot.submit_begin_ns,
      .width = snapshot.request.width,
  };
}

} // namespace

void publish_final(VulkanResidencyPersistentRun &run) noexcept {
  PersistentResidencySlidingFinalCompletion completion = nullptr;
  void *user = nullptr;
  PersistentResidencySlidingFinal final{};
  PersistentResidencySlidingControl *control_pointer = nullptr;
  {
    std::lock_guard lock{run.gate};
    control_pointer = run.control;
  }
  if (control_pointer == nullptr) {
    return;
  }
  {
    PersistentResidencySlidingControl &control = *control_pointer;
    std::lock_guard control_lock{control.gate};
    std::lock_guard lock{run.gate};
    if (!run.active || run.final_sent || run.control != control_pointer) {
      return;
    }
    if (control.final_callback_count != 0u) {
      return;
    }
    const FinalSnapshot snapshot{
        .request = run.request,
        .submit_begin_ns = run.submit_begin_ns,
        .physical_terminal = run.quarantined
                                  ? NativeTerminal::UnknownMayWrite
                                  : NativeTerminal::Known,
    };
    if (snapshot.physical_terminal == NativeTerminal::UnknownMayWrite) {
      persistent_sliding_mark_unknown_locked(control);
    }
    const bool unknown = control.quarantined;
    const bool success = !unknown && !control.service_failed &&
                         control.failed_admission_count == 0u &&
                         control.first_failure.ok;
    const std::uint64_t end = control.accepted_end;
    if (!unknown && control.next_acknowledgement_coordinate != end) {
      return;
    }
    if (!unknown) {
      clear_active(run);
    }
    ++control.final_callback_count;
    final = PersistentResidencySlidingFinal{
        .check =
            success
                ? rund::AccelCheck{true, "ok"}
                : (unknown ? rund::AccelCheck{false, "compute_device_lost"}
                           : (control.first_failure.ok
                                  ? rund::AccelCheck{false,
                                                     "compute_backend_failed"}
                                  : control.first_failure)),
        .terminal =
            unknown ? NativeTerminal::UnknownMayWrite : NativeTerminal::Known,
        .evidence = evidence(snapshot, control, success, unknown),
    };
    completion = snapshot.request.final;
    user = snapshot.request.user;
    run.final_sent = true;
    if (!unknown) {
      control.native.reset();
      control.active = false;
      run.request = {};
      run.control = nullptr;
      run.roles = {};
      run.tail = {};
      run.cells = {};
      run.active = false;
    } else {
      run.request.final = nullptr;
      run.request.user = nullptr;
      run.control = nullptr;
      control.active = false;
    }
  }
  if (completion != nullptr && user != nullptr) {
    completion(user, std::move(final));
  }
}

void close_known(VulkanResidencyPersistentRun &run) noexcept {
  PersistentResidencySlidingControl *control = nullptr;
  VulkanPipeline *pipeline = nullptr;
  VulkanTimelinePoint terminal{};
  {
    std::lock_guard lock{run.gate};
    if (!run.active || run.final_sent || run.request.coordinate_count == 0u) {
      return;
    }
    control = run.control;
  }
  if (control == nullptr) {
    return;
  }
  std::unique_lock control_lock{control->gate};
  std::unique_lock run_lock{run.gate};
  if (!run.active || run.final_sent || run.control != control) {
    return;
  }
  const std::uint64_t end = control->accepted_end;
  if (end == 0u || end > run.request.coordinate_count) {
    run.quarantined = true;
    persistent_sliding_mark_unknown_locked(*control);
    run_lock.unlock();
    control_lock.unlock();
    publish_final(run);
    return;
  }
  pipeline = native_role(run, end - 1u).pipeline;
  terminal = point(run, end - 1u);
  const bool closed =
      pipeline != nullptr &&
      CloseVulkanTimelineGeneration(pipeline->adapter->timeline, terminal).ok;
  if (!closed) {
    run.quarantined = true;
    if (pipeline != nullptr && pipeline->adapter != nullptr) {
      pipeline->adapter->residency_quarantined.store(true,
                                                     std::memory_order_release);
    }
    persistent_sliding_mark_unknown_locked(*control);
  }
  run_lock.unlock();
  control_lock.unlock();
  publish_final(run);
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
