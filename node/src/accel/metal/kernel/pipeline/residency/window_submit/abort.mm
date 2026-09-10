#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
AbortMetalResidencyWindow(const std::shared_ptr<void> &prepared,
                          const BackendResidencyWindowAbort &abort) noexcept {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    auto *const leader = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(leader) || leader->adapter == nullptr ||
        abort.failure.ok || abort.failure.reason == nullptr ||
        abort.failure.reason[0u] == '\0' || abort.plan_identity == 0u ||
        abort.token == 0u || abort.generation == 0u) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    MetalAdapter &adapter = *leader->adapter;
    std::unique_lock terminal_gate{adapter.residency_terminal_gate};
    MetalResidencyWindowRun &run = leader->residency_window.run;
    std::lock_guard run_lock{run.gate};
    if (!run.active || run.final_sent ||
        abort.plan_identity != run.request.plan_identity ||
        abort.token != run.request.token ||
        abort.generation != run.request.generation) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    std::size_t opened = 0u;
    const std::uint64_t now = MonotonicNanoseconds();
    for (std::size_t index = 0u; index < run.request.batch_count; ++index) {
      auto *const sequence = static_cast<MetalSequence *>(
          run.request.batches[index].prepared.get());
      if (!ValidMetalSequence(sequence) || sequence->adapter != &adapter) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      MetalResidencyWindowCommand *command = nullptr;
      for (MetalResidencyWindowCommand &candidate :
           sequence->residency_window.commands) {
        const std::shared_ptr<void> candidate_leader = candidate.leader.lock();
        if (candidate_leader.get() == leader &&
            candidate.batch_index == index) {
          command = &candidate;
          break;
        }
      }
      if (command == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      if (command->terminal.state() != TerminalState::Armed ||
          command->watchdog_deadline_ns.load(std::memory_order_acquire) != 0u) {
        continue;
      }
      void *const guard = [sequence->guard_zero contents];
      if (guard == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      command->admission = abort.failure;
      command->force_abort_unknown = true;
      *static_cast<std::uint32_t *>(guard) = 1u;
      run.native_inflight =
          std::min<std::uint64_t>(2u, run.native_inflight + 1u);
      run.native_inflight_peak =
          std::max(run.native_inflight_peak, run.native_inflight);
      const std::uint64_t deadline =
          now > std::numeric_limits<std::uint64_t>::max() -
                      MetalResidencyWindowFaultDeadlineNs
              ? std::numeric_limits<std::uint64_t>::max()
              : now + MetalResidencyWindowFaultDeadlineNs;
      command->watchdog_deadline_ns.store(deadline, std::memory_order_release);
      dispatch_source_set_timer(
          command->watchdog,
          dispatch_time(
              DISPATCH_TIME_NOW,
              static_cast<int64_t>(MetalResidencyWindowFaultDeadlineNs)),
          DISPATCH_TIME_FOREVER, 0u);
      sequence->residency_window.ready.signaledValue = command->ready_value;
      ++opened;
    }
    if (opened == 0u) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    adapter.residency_quarantined.store(true, std::memory_order_release);
    leader->residency_window.phase = MetalResidencyWindowPhase::Quarantined;
    SetMetalLastError(adapter, "compute_device_lost");
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(abort);
#endif
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
