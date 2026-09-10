#include "internal.hpp"

#include "../encode.hpp"

#include "../../../../../clock.hpp"

#include <limits>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck SubmitMetalResidencyWindow(
    const BackendResidencyWindowRequest &request) noexcept {
  const rund::AccelCheck unavailable{false, "accel_metal_command_unavailable"};
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (request.plan_identity == 0u || request.token == 0u ||
        request.generation == 0u || request.batch_count == 0u ||
        request.batch_count > ResidencyWindowCapacity ||
        request.release == nullptr || request.final == nullptr ||
        request.user == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    auto *const leader =
        static_cast<MetalSequence *>(request.batches[0u].prepared.get());
    if (!ValidMetalSequence(leader) || leader->adapter == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    MetalAdapter &adapter = *leader->adapter;
    std::unique_lock terminal_gate{adapter.residency_terminal_gate};
    if (adapter.residency_quarantined.load(std::memory_order_acquire)) {
      return rund::AccelCheck{false, "compute_device_lost"};
    }
    id<MTL4CommandQueue> const queue = MetalResidencyQueue(*leader);
    if (queue == nil) {
      return unavailable;
    }
    std::array<MetalSequence *, ResidencyWindowCapacity> sequences{};
    std::array<std::size_t, ResidencyWindowCapacity> slots{};
    std::array<std::uint64_t, ResidencyWindowCapacity> values{};
    std::size_t encoded_count = 0u;
    for (std::size_t index = 0u; index < request.batch_count; ++index) {
      const BackendResidencyWindowBatch &batch = request.batches[index];
      auto *const sequence = static_cast<MetalSequence *>(batch.prepared.get());
      if (!ValidMetalSequence(sequence) || sequence->adapter != &adapter ||
          sequence->direct_aggregate || sequence->state_count != 0u ||
          sequence->guard_zero == nil ||
          [sequence->guard_zero contents] == nullptr ||
          sequence->residency_window.service == nil ||
          sequence->residency_submission.terminal.active() ||
          !sequence->residency_window.ready_for_submit() ||
          request.first_epoch >
              std::numeric_limits<std::uint64_t>::max() - index ||
          batch.epoch != request.first_epoch + index ||
          batch.bank != batch.epoch % 2u || batch.control_generation == 0u ||
          batch.local_count == 0u ||
          batch.local_count > ResidencyWindowLocalCapacity) {
        for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
          CancelMetalResidencyWindowCommand(
              sequences[prior]->residency_window.commands[slots[prior]]);
        }
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      // The bounded window is double-buffered by bank. A bank owns one
      // prepared command stream for the entire window, and the two adjacent
      // banks must not alias the same no-write gate. This makes a later
      // admission unable to overwrite an earlier in-flight batch's gate.
      if ((index >= 2u && sequence != sequences[index - 2u]) ||
          (index != 0u && sequence == sequences[index - 1u])) {
        for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
          CancelMetalResidencyWindowCommand(
              sequences[prior]->residency_window.commands[slots[prior]]);
        }
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      {
        std::lock_guard submission_lock{sequence->submission.mutex};
        if (sequence->submission.active()) {
          for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
            CancelMetalResidencyWindowCommand(
                sequences[prior]->residency_window.commands[slots[prior]]);
          }
          return rund::AccelCheck{false, "compute_pipeline_busy"};
        }
      }
      std::size_t slot = 0u;
      for (std::size_t prior = 0u; prior < index; ++prior) {
        slot += sequences[prior] == sequence ? 1u : 0u;
      }
      if (slot >= sequence->residency_window.commands.size() ||
          sequence->residency_window.commands[slot].terminal.active() ||
          sequence->residency_window.event_value ==
              TerminalCell::MaxGeneration) {
        for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
          CancelMetalResidencyWindowCommand(
              sequences[prior]->residency_window.commands[slots[prior]]);
        }
        return rund::AccelCheck{false, "compute_pipeline_busy"};
      }
      const rund::AccelCheck prepared = PrepareMetalResidencyWindowCommand(
          *sequence, sequence->residency_window.commands[slot],
          std::span<const std::uint32_t>{batch.locals.data(),
                                         batch.local_count});
      if (!prepared.ok) {
        for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
          CancelMetalResidencyWindowCommand(
              sequences[prior]->residency_window.commands[slots[prior]]);
        }
        return prepared;
      }
      sequences[index] = sequence;
      slots[index] = slot;
      values[index] = ++sequence->residency_window.event_value;
      ++encoded_count;
    }

    MetalResidencyWindowRun &run = leader->residency_window.run;
    std::unique_lock run_lock{run.gate, std::try_to_lock};
    if (!run_lock.owns_lock() || run.active) {
      for (std::size_t index = 0u; index < encoded_count; ++index) {
        CancelMetalResidencyWindowCommand(
            sequences[index]->residency_window.commands[slots[index]]);
      }
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
    const bool force_device_lost =
        adapter.device_loss_fault.take(SubmitKind::Work);
    const bool force_terminal_loss =
        adapter.fault_residency_terminal_once.exchange(
            false, std::memory_order_acq_rel);
    const std::weak_ptr<void> leader_owner{request.batches[0u].prepared};
    for (std::size_t index = 0u; index < request.batch_count; ++index) {
      MetalResidencyWindowCommand &command =
          sequences[index]->residency_window.commands[slots[index]];
      command.leader = leader_owner;
      command.terminal_generation = values[index];
      command.ready_value = values[index];
      command.done_value = values[index];
      command.batch_index = index;
      command.force_device_lost =
          force_device_lost && index + 1u == request.batch_count;
      command.force_terminal_loss =
          force_terminal_loss && index + 1u == request.batch_count;
      if (!command.terminal.arm(values[index])) {
        for (std::size_t prior = 0u; prior < encoded_count; ++prior) {
          MetalResidencyWindowCommand &armed =
              sequences[prior]->residency_window.commands[slots[prior]];
          if (armed.terminal.state() == TerminalState::Armed) {
            static_cast<void>(armed.terminal.resolve(armed.terminal_generation,
                                                     NativeTerminal::Known));
            static_cast<void>(armed.terminal.release(armed.terminal_generation,
                                                     NativeTerminal::Known));
          }
          CancelMetalResidencyWindowCommand(armed);
        }
        return unavailable;
      }
    }
    run.request = request;
    run.receipts = {};
    run.results = {};
    run.completed = {};
    run.release_count = 0u;
    run.native_inflight = 0u;
    run.native_inflight_peak = 0u;
    run.submit_begin_ns = MonotonicNanoseconds();
    run.active = true;
    run.unknown = false;
    run.final_sent = false;

    for (std::size_t index = 0u; index < request.batch_count; ++index) {
      MetalSequence &sequence = *sequences[index];
      MetalResidencyWindowCommand &command =
          sequence.residency_window.commands[slots[index]];
      const std::weak_ptr<void> sequence_owner{request.batches[index].prepared};
      const std::uint64_t done_value = command.done_value;
      const std::uint64_t terminal_generation = command.terminal_generation;
      const std::size_t command_slot = slots[index];
      if (!command.force_terminal_loss) {
        [sequence.residency_window.done
            notifyListener:sequence.residency_window.listener
                   atValue:done_value
                     block:^(id<MTLSharedEvent>, std::uint64_t value) {
                       const std::shared_ptr<void> retained =
                           sequence_owner.lock();
                       if (retained == nullptr || value < done_value) {
                         return;
                       }
                       CompleteMetalResidencyWindowCommand(
                           *static_cast<MetalSequence *>(retained.get()),
                           command_slot, terminal_generation,
                           NativeTerminal::Known);
                     }];
      }
      [queue waitForEvent:sequence.residency_window.ready
                    value:command.ready_value];
      const id<MTL4CommandBuffer> commands[] = {command.command};
      [queue commit:commands count:1u];
      if (!command.force_terminal_loss) {
        [queue signalEvent:sequence.residency_window.done
                     value:command.done_value];
      }
    }
    run_lock.unlock();
    terminal_gate.unlock();
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(request);
#endif
  return unavailable;
}

#endif

} // namespace rund::node::accel::detail
