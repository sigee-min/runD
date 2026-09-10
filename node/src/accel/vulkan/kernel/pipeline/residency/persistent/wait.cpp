#include "../mode.hpp"
#include "../generated_indirect/map.hpp"
#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck WaitVulkanResidencyPersistentDone(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingDoneWait &wait,
    PersistentResidencySlidingDoneObservation &observation) noexcept {
  using namespace vulkan_persistent_detail;
  observation = {};
  std::unique_lock control_lock{control.gate};
  VulkanResidencyPersistentRun *const run = active_run(control);
  if (run == nullptr || !control.active || control.quarantined ||
      wait.identity.plan_identity != control.plan_identity ||
      wait.identity.token != control.token ||
      wait.identity.generation != control.generation ||
      wait.identity.coordinate != control.next_wait_coordinate ||
      wait.identity.coordinate >= persistent_sliding_accepted_end(control) ||
      wait.identity.coordinate >= control.next_ready_coordinate) {
    return invalid();
  }
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(control, wait.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, wait.identity)) {
    return invalid();
  }
  const std::uint64_t coordinate = wait.identity.coordinate;
  VulkanPipeline *pipeline = nullptr;
  std::shared_ptr<void> pipeline_keep;
  VulkanTimelinePoint terminal{};
  rund::AccelCheck admission{true, "ok"};
  std::uint64_t dispatch_count = 0u;
  std::uint64_t control_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  {
    std::lock_guard run_lock{run->gate};
    const std::size_t cell_index =
        static_cast<std::size_t>(coordinate % VulkanResidencyWindowCapacity);
    VulkanResidencyPersistentCell &cell = run->cells[cell_index];
    pipeline = native_role(*run, coordinate).pipeline;
    if (!run->active || run->final_sent || run->quarantined ||
        pipeline == nullptr || !cell.signaled || cell.waiting || cell.observed ||
        cell.coordinate != coordinate) {
      return invalid();
    }
    admission = cell.admission;
    pipeline_keep = source_role(*run, coordinate).prepared;
    const VulkanResidencyPersistentRole &native =
        native_role(*run, coordinate);
    dispatch_count = native.dispatch_count;
    control_count = native.control_count;
    reset_count = native.reset_count;
    reset_bytes = native.reset_bytes;
    terminal = point(*run, coordinate);
  }
  if (pipeline == nullptr || pipeline_keep == nullptr ||
      pipeline->residency == nullptr) {
    return invalid();
  }
  const auto access =
      vulkan_generated_indirect_detail::map_access(*pipeline->residency);
  if (!access.usable()) {
    return invalid();
  }
  {
    std::lock_guard run_lock{run->gate};
    const std::size_t cell_index = static_cast<std::size_t>(
        coordinate % VulkanResidencyWindowCapacity);
    VulkanResidencyPersistentCell &cell = run->cells[cell_index];
    if (!run->active || run->final_sent || run->quarantined ||
        cell.coordinate != coordinate || cell.waiting || cell.observed) {
      return invalid();
    }
    cell.waiting = true;
  }
  control_lock.unlock();
  const rund::AccelCheck waited =
      WaitVulkanTimelineDone(pipeline->adapter->timeline, terminal, TimeoutNs);
  KernelResult result{};
  NativeTerminal terminal_kind = NativeTerminal::Known;
  bool known_gate_failure = false;
  const char *gate_failure_reason = "accel_kernel_pipeline_invalid";
  if (!waited.ok) {
    result.check = {false, "compute_device_lost"};
    terminal_kind = NativeTerminal::UnknownMayWrite;
  } else {
    std::atomic_thread_fence(std::memory_order_acquire);
    const bool accepted = gate_result(*pipeline->residency,
                                      wait.identity.descriptor_generation,
                                      known_gate_failure, gate_failure_reason);
    if (!accepted) {
      if (known_gate_failure) {
        result.check = admission.ok
                           ? rund::AccelCheck{false, gate_failure_reason}
                           : admission;
        terminal_kind = NativeTerminal::Known;
      } else {
        result.check = {false, "accel_kernel_pipeline_invalid"};
        terminal_kind = NativeTerminal::UnknownMayWrite;
      }
    } else {
      result = ObserveVulkanResidency(
          *pipeline, admission, dispatch_count, control_count, reset_count,
          reset_bytes,
          wait.identity.control_generation);
      terminal_kind = result.terminal;
    }
  }
  const bool unknown = terminal_kind == NativeTerminal::UnknownMayWrite;
  observation = PersistentResidencySlidingDoneObservation{
      .identity = wait.identity,
      .check = result.check,
      .terminal = terminal_kind,
      .dispatched = admission.ok,
      .completed = !unknown,
      .may_write = unknown || admission.ok,
  };
  bool emit_unknown = false;
  {
    control_lock.lock();
    VulkanResidencyPersistentRun *const current = active_run(control);
    PersistentResidencySlidingServiceIdentity current_expected{};
    const bool identity_ok =
        current == run && control.active && !control.quarantined &&
        wait.identity.plan_identity == control.plan_identity &&
        wait.identity.token == control.token &&
        wait.identity.generation == control.generation &&
        wait.identity.coordinate == control.next_wait_coordinate &&
        wait.identity.coordinate <
            persistent_sliding_accepted_end(control) &&
        wait.identity.coordinate < control.next_ready_coordinate &&
        persistent_sliding_service_identity(
            control, wait.identity.coordinate, current_expected) &&
        persistent_sliding_same_service_identity(current_expected,
                                                  wait.identity);
    std::lock_guard run_lock{run->gate};
    if (!identity_ok || !run->active || run->final_sent ||
        run->quarantined) {
      const std::size_t cell_index = static_cast<std::size_t>(
          coordinate % VulkanResidencyWindowCapacity);
      VulkanResidencyPersistentCell &cell = run->cells[cell_index];
      if (cell.coordinate == coordinate && cell.waiting) {
        cell.waiting = false;
      }
      run->quarantined = true;
      if (pipeline->residency != nullptr) {
        pipeline->residency->sliding.quarantined.store(
            true, std::memory_order_release);
      }
      if (pipeline->adapter != nullptr) {
        pipeline->adapter->residency_quarantined.store(
            true, std::memory_order_release);
      }
      persistent_sliding_mark_unknown_locked(control);
      emit_unknown = true;
    } else {
      const std::size_t cell_index = static_cast<std::size_t>(
          coordinate % VulkanResidencyWindowCapacity);
      VulkanResidencyPersistentCell &cell = run->cells[cell_index];
      if (cell.coordinate != coordinate || !cell.waiting || cell.observed) {
        if (cell.coordinate == coordinate && cell.waiting) {
          cell.waiting = false;
        }
        run->quarantined = true;
        if (pipeline->residency != nullptr) {
          pipeline->residency->sliding.quarantined.store(
              true, std::memory_order_release);
        }
        if (pipeline->adapter != nullptr) {
          pipeline->adapter->residency_quarantined.store(
              true, std::memory_order_release);
        }
        persistent_sliding_mark_unknown_locked(control);
        emit_unknown = true;
      } else {
        cell.waiting = false;
        cell.observed = true;
        if (!result.check.ok && control.first_failure.ok) {
          control.first_failure = result.check;
        }
        if (!admission.ok || (known_gate_failure && admission.ok)) {
          if (control.suppressed_count == 0u) {
            control.suppressed_first = coordinate;
          }
          ++control.suppressed_count;
        }
        if (unknown) {
          run->quarantined = true;
          if (pipeline->residency != nullptr) {
            pipeline->residency->sliding.quarantined.store(
                true, std::memory_order_release);
          }
          if (pipeline->adapter != nullptr) {
            pipeline->adapter->residency_quarantined.store(
                true, std::memory_order_release);
          }
          persistent_sliding_mark_unknown_locked(
              control, result.check.ok
                            ? rund::AccelCheck{false, "compute_device_lost"}
                            : result.check);
          emit_unknown = true;
        }
        ++control.next_wait_coordinate;
        ++control.next_observation_coordinate;
        control.gpu_completed_coordinates =
            control.next_observation_coordinate;
        control.completed_prefix = result.check.ok
                                       ? control.next_observation_coordinate
                                       : coordinate;
        ++control.backing_wait_count;
      }
    }
  }
  control_lock.unlock();
  if (emit_unknown) {
    publish_final(*run);
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
