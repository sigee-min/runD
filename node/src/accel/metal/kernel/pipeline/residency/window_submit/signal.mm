#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck SignalMetalResidencyWindow(
    const std::shared_ptr<void> &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    auto *const sequence = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(sequence) || sequence->adapter == nullptr ||
        signal.plan_identity == 0u || signal.token == 0u ||
        signal.generation == 0u) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    MetalAdapter &adapter = *sequence->adapter;
    std::unique_lock terminal_gate{adapter.residency_terminal_gate};
    MetalResidencyWindowCommand *command = nullptr;
    std::shared_ptr<void> leader_owner{};
    for (MetalResidencyWindowCommand &candidate :
         sequence->residency_window.commands) {
      const std::shared_ptr<void> candidate_leader = candidate.leader.lock();
      auto *const candidate_owner =
          static_cast<MetalSequence *>(candidate_leader.get());
      if (candidate_owner != nullptr &&
          signal.epoch >=
              candidate_owner->residency_window.run.request.first_epoch &&
          candidate.batch_index ==
              signal.epoch -
                  candidate_owner->residency_window.run.request.first_epoch) {
        command = &candidate;
        leader_owner = candidate_leader;
        break;
      }
    }
    auto *const leader = static_cast<MetalSequence *>(leader_owner.get());
    if (command == nullptr || leader == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    MetalResidencyWindowRun &run = leader->residency_window.run;
    std::lock_guard run_lock{run.gate};
    if (!run.active || run.final_sent ||
        signal.plan_identity != run.request.plan_identity ||
        signal.token != run.request.token ||
        signal.generation != run.request.generation ||
        signal.epoch < run.request.first_epoch ||
        signal.epoch - run.request.first_epoch >= run.request.batch_count ||
        signal.control_generation !=
            run.request.batches[signal.epoch - run.request.first_epoch]
                .control_generation ||
        signal.bank !=
            run.request.batches[signal.epoch - run.request.first_epoch].bank ||
        run.request.batches[signal.epoch - run.request.first_epoch]
                .prepared.get() != prepared.get() ||
        command->terminal.state() != TerminalState::Armed ||
        command->watchdog_deadline_ns.load(std::memory_order_acquire) != 0u ||
        run.native_inflight >= 2u) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    void *const guard = [sequence->guard_zero contents];
    if (guard == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (!signal.admission.ok && (signal.admission.reason == nullptr ||
                                 signal.admission.reason[0u] == '\0')) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    if (signal.admission.ok) {
      void *const control = [sequence->control contents];
      if (control == nullptr) {
        return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
      }
      const PreparedPipelineControl initial{.generation =
                                                signal.control_generation - 1u};
      std::memcpy(control, &initial, sizeof(initial));
    }
    command->admission = signal.admission;
    *static_cast<std::uint32_t *>(guard) = signal.admission.ok ? 0u : 1u;
    ++run.native_inflight;
    run.native_inflight_peak =
        std::max(run.native_inflight_peak, run.native_inflight);
    const std::uint64_t deadline_delta =
        command->force_terminal_loss ? MetalResidencyWindowFaultDeadlineNs
                                     : MetalResidencyWindowTerminalDeadlineNs;
    const std::uint64_t now = MonotonicNanoseconds();
    const std::uint64_t deadline =
        now > std::numeric_limits<std::uint64_t>::max() - deadline_delta
            ? std::numeric_limits<std::uint64_t>::max()
            : now + deadline_delta;
    command->watchdog_deadline_ns.store(deadline, std::memory_order_release);
    dispatch_source_set_timer(
        command->watchdog,
        dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(deadline_delta)),
        DISPATCH_TIME_FOREVER, 0u);
    sequence->residency_window.ready.signaledValue = command->ready_value;
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(signal);
#endif
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
