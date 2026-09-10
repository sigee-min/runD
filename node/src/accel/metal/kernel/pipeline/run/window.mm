#include "../state.hpp"
#include "internal.hpp"

#include "../residency/local.hpp"

#include "../../../../clock.hpp"
#include <array>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
namespace {

void DeliverMetalResidencyWindowCallbacks(void *const raw) noexcept {
  auto *const delivery = static_cast<MetalResidencyWindowDelivery *>(raw);
  if (delivery == nullptr || delivery->callbacks.release == nullptr ||
      delivery->callbacks.final == nullptr ||
      delivery->callbacks.user == nullptr) {
    return;
  }
  BackendResidencyWindowRequest callbacks = std::move(delivery->callbacks);
  auto releases = std::move(delivery->releases);
  const std::size_t release_count = delivery->release_count;
  BackendResidencyWindowFinal final = std::move(delivery->final);
  const bool emit_final = delivery->emit_final;
  delivery->release_count = 0u;
  delivery->emit_final = false;
  for (std::size_t index = 0u; index < release_count; ++index) {
    callbacks.release(callbacks.user, std::move(releases[index]));
  }
  if (emit_final) {
    callbacks.final(callbacks.user, std::move(final));
  }
}

} // namespace

void CompleteMetalResidencyWindowCommand(
    MetalSequence &sequence, const std::size_t slot,
    const std::uint64_t generation, const NativeTerminal terminal) noexcept {
  if (sequence.adapter == nullptr ||
      slot >= sequence.residency_window.commands.size()) {
    return;
  }
  MetalAdapter &adapter = *sequence.adapter;
  std::unique_lock terminal_gate{adapter.residency_terminal_gate};
  MetalResidencyWindowCommand &command =
      sequence.residency_window.commands[slot];
  const NativeTerminal effective_terminal =
      command.force_abort_unknown ? NativeTerminal::UnknownMayWrite : terminal;
  const std::shared_ptr<void> leader_owner = command.leader.lock();
  auto *const leader = static_cast<MetalSequence *>(leader_owner.get());
  if (leader == nullptr || command.terminal_generation != generation ||
      !command.terminal.resolve(generation, effective_terminal)) {
    return;
  }
  command.watchdog_deadline_ns.store(0u, std::memory_order_release);
  dispatch_source_set_timer(command.watchdog, DISPATCH_TIME_FOREVER,
                            DISPATCH_TIME_FOREVER, 0u);
  const bool known = effective_terminal == NativeTerminal::Known;
  if (known) {
    [command.allocator reset];
    void *const guard = [sequence.guard_zero contents];
    if (guard == nullptr) {
      command.admission =
          rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    } else {
      *static_cast<std::uint32_t *>(guard) = 0u;
    }
  } else {
    adapter.residency_quarantined.store(true, std::memory_order_release);
    sequence.residency_window.phase = MetalResidencyWindowPhase::Quarantined;
    leader->residency_window.phase = MetalResidencyWindowPhase::Quarantined;
    SetMetalLastError(adapter, "compute_device_lost");
  }
  terminal_gate.unlock();

  KernelResult result{
      .check = rund::AccelCheck{true, "ok"},
      .stats =
          rund::RuntimeStats{
              .run =
                  {
                      .work =
                          {
                              .dispatch_count = command.dispatch_count,
                              .command_submit_count = 1u,
                              .reset_command_count = command.reset_count,
                              .reset_bytes = command.reset_bytes,
                          },
                  },
              .outcome = {.ok = true, .reason = "ok"},
          },
      .pipeline = {.control_command_count = command.control_count,
                   .submitted = true},
      .terminal = effective_terminal,
  };
  if (!known || command.force_device_lost) {
    result.check = rund::AccelCheck{false, "compute_device_lost"};
  } else if (!command.admission.ok) {
    result.check = command.admission;
  } else if (!ObserveMetalControl(sequence, result.pipeline) ||
             !ObserveMetalProfile(sequence, result.pipeline)) {
    result.check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  result.stats.outcome.ok = result.check.ok;
  result.stats.outcome.reason = result.check.reason;
  if (result.check.ok) {
    RecordMetalDispatches(adapter, command.dispatch_count);
  } else {
    result.stats.run.work.dispatch_count = 0u;
    result.stats.run.work.reset_command_count = 0u;
    result.stats.run.work.reset_bytes = 0u;
  }
  RecordMetalCommandSubmitWaitNs(
      adapter, known ? MonotonicNanoseconds() -
                           leader->residency_window.run.submit_begin_ns
                     : 0u);

  std::array<BackendResidencyWindowRelease, ResidencyWindowCapacity> releases{};
  std::size_t release_count = 0u;
  BackendResidencyWindowFinal final{};
  bool emit_final = false;
  BackendResidencyWindowRequest callbacks{};
  {
    MetalResidencyWindowRun &run = leader->residency_window.run;
    std::lock_guard run_lock{run.gate};
    const std::size_t batch_index = command.batch_index;
    if (!run.active || run.final_sent ||
        batch_index >= run.request.batch_count || run.completed[batch_index] ||
        run.request.batches[batch_index].prepared.get() != &sequence ||
        run.request.first_epoch >
            std::numeric_limits<std::uint64_t>::max() - batch_index ||
        run.request.batches[batch_index].epoch !=
            run.request.first_epoch + batch_index) {
      return;
    }
    if (known && result.pipeline.control_observed &&
        result.pipeline.control.generation !=
            run.request.batches[batch_index].control_generation) {
      result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      result.stats.outcome.ok = false;
      result.stats.outcome.reason = result.check.reason;
      result.stats.run.work.dispatch_count = 0u;
    }
    if (known) {
      if (run.native_inflight == 0u) {
        result.check = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
        result.stats.outcome.ok = false;
        result.stats.outcome.reason = result.check.reason;
      } else {
        --run.native_inflight;
      }
    } else {
      run.unknown = true;
    }
    result.stats.run.work.command_inflight_peak = run.native_inflight_peak;
    run.receipts[batch_index] = BackendResidencyWindowReceipt{
        .check = result.check,
        .terminal = effective_terminal,
        .epoch = run.request.batches[batch_index].epoch,
        .backend_sequence = run.request.batches[batch_index].epoch + 1u,
        .bank = run.request.batches[batch_index].bank,
        .dispatched = true,
        .completed = known,
        .may_write = !known || command.admission.ok,
    };
    run.results[batch_index] = result;
    run.completed[batch_index] = true;
    while (run.release_count < run.request.batch_count &&
           run.completed[run.release_count]) {
      const std::size_t index = run.release_count++;
      releases[release_count++] = BackendResidencyWindowRelease{
          .receipt = run.receipts[index],
          .result = run.results[index],
      };
    }
    callbacks = run.request;
    if (run.release_count == run.request.batch_count) {
      run.final_sent = true;
      const BackendResidencyWindowReceipt *failure = nullptr;
      for (std::size_t index = 0u; index < run.request.batch_count; ++index) {
        if (failure == nullptr && !run.receipts[index].check.ok) {
          failure = &run.receipts[index];
        }
      }
      final = BackendResidencyWindowFinal{
          .check = failure == nullptr ? rund::AccelCheck{true, "ok"}
                                      : failure->check,
          .terminal = run.unknown ? NativeTerminal::UnknownMayWrite
                                  : NativeTerminal::Known,
          .plan_identity = run.request.plan_identity,
          .token = run.request.token,
          .generation = run.request.generation,
          .first_epoch = run.request.first_epoch,
          .public_handoffs = 1u,
          .native_batches = run.request.batch_count,
          .queue_calls = run.request.batch_count,
          .native_inflight_peak = run.native_inflight_peak,
          .receipts = run.receipts,
          .receipt_count = run.request.batch_count,
          .completed_ns = MonotonicNanoseconds(),
      };
      emit_final = true;
      if (!run.unknown) {
        run.active = false;
        run.request = {};
      } else {
        // Unknown retains only the backend/prepared lifetime quarantine. The
        // common callback control is caller-owned and may die immediately
        // after Final, so no dormant callback/user pointer may survive it.
        run.request.release = nullptr;
        run.request.final = nullptr;
        run.request.user = nullptr;
      }
    }
  }
  if (known) {
    static_cast<void>(command.terminal.release(generation, effective_terminal));
  }
  if (release_count != 0u || emit_final) {
    MetalResidencyWindowDelivery &delivery =
        sequence.residency_window.deliveries[slot];
    delivery.callbacks = std::move(callbacks);
    delivery.releases = std::move(releases);
    delivery.release_count = release_count;
    delivery.final = std::move(final);
    delivery.emit_final = emit_final;
    dispatch_async_f(leader->residency_window.service, &delivery,
                     DeliverMetalResidencyWindowCallbacks);
  }
}
#endif


#endif

} // namespace rund::node::accel::detail
