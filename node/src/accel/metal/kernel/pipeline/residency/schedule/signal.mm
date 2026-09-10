#include "internal.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace schedule_detail {

bool signal_locked(const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
                  MetalResidencyScheduleCommand &entry,
                  const rund::AccelCheck admission) noexcept {
  if (entry.signaled || entry.completed || entry.sequence == nullptr ||
      owner->native_inflight >= 2u ||
      (entry.epoch >= 2u && !owner->commands[entry.epoch - 2u].completed)) {
    return false;
  }
  void *const guard = [entry.sequence->guard_zero contents];
  if (guard == nullptr ||
      (!admission.ok &&
       (admission.reason == nullptr || admission.reason[0u] == '\0'))) {
    return false;
  }
  if (admission.ok) {
    void *const control = [entry.sequence->control contents];
    if (control == nullptr || entry.control_generation == 0u) {
      return false;
    }
    const PreparedPipelineControl initial{.generation =
                                              entry.control_generation - 1u};
    std::memcpy(control, &initial, sizeof(initial));
  }
  entry.admission = admission;
  *static_cast<std::uint32_t *>(guard) = admission.ok ? 0u : 1u;
  entry.signaled = true;
  ++owner->native_inflight;
  owner->native_inflight_peak =
      std::max(owner->native_inflight_peak, owner->native_inflight);
  static_cast<void>(BeginMetalResidencyCommand(*owner->adapter));
  entry.sequence->residency_window.ready.signaledValue = entry.ready_value;
  arm_timeout(owner, entry.epoch,
              entry.force_terminal_loss ? FaultTerminalDeadlineNs
                                        : TerminalDeadlineNs);
  return true;
}

} // namespace schedule_detail

#endif

#endif

rund::AccelCheck SignalMetalResidencySchedule(
    const std::shared_ptr<void> &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    auto *const sequence = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(sequence) || sequence->adapter == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    const std::shared_ptr<schedule_detail::MetalResidencyScheduleOwner> owner =
        schedule_detail::owner_of(sequence->residency_schedule.lock());
    if (owner == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    std::scoped_lock lock{sequence->adapter->residency_terminal_gate,
                          owner->gate};
    if (!owner->active || owner->final_sent ||
        signal.plan_identity != owner->request.plan_identity ||
        signal.token != owner->request.token ||
        signal.generation != owner->request.generation ||
        signal.epoch >= owner->commands.size()) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    schedule_detail::MetalResidencyScheduleCommand &entry =
        owner->commands[signal.epoch];
    if (entry.sequence != sequence || signal.bank != signal.epoch % 2u ||
        signal.control_generation != entry.control_generation ||
        !schedule_detail::signal_locked(owner, entry, signal.admission)) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(signal);
#endif
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

} // namespace rund::node::accel::detail
