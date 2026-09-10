#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace rund::node::accel::detail::schedule_detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace {

void clear_active_sequences(MetalResidencyScheduleOwner &owner) noexcept {
  for (std::size_t role = 0u; role < owner.request.role_count; ++role) {
    auto *const sequence =
        static_cast<MetalSequence *>(owner.request.roles[role].prepared.get());
    if (sequence != nullptr) {
      sequence->residency_schedule.reset();
    }
  }
}

} // namespace

void deliver(void *const raw) noexcept {
  auto *const delivery = static_cast<MetalResidencyScheduleDelivery *>(raw);
  if (delivery == nullptr || delivery->owner == nullptr ||
      delivery->callbacks.release == nullptr ||
      delivery->callbacks.final == nullptr ||
      delivery->callbacks.user == nullptr) {
    return;
  }
  BackendResidencyScheduleRequest callbacks = delivery->callbacks;
  BackendResidencyScheduleRelease release = std::move(delivery->release);
  BackendResidencyScheduleFinal final = std::move(delivery->final);
  const bool emit_final = delivery->emit_final;
  callbacks.release(callbacks.user, std::move(release));
  if (emit_final) {
    callbacks.final(callbacks.user, std::move(final));
  }
  delivery->callbacks = {};
  delivery->emit_final = false;
  delivery->owner.reset();
}

void arm_timeout(const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
                 const std::uint64_t epoch, const std::uint64_t delay) noexcept {
  const std::weak_ptr<MetalResidencyScheduleOwner> weak{owner};
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(delay)),
                 dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0u), ^{
                   const std::shared_ptr<MetalResidencyScheduleOwner> retained =
                       weak.lock();
                   if (retained != nullptr) {
                     complete(retained, epoch, NativeTerminal::UnknownMayWrite);
                   }
                 });
}

void complete(const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
              const std::uint64_t epoch,
              const NativeTerminal terminal) noexcept {
  if (owner == nullptr || owner->magic != ScheduleMagic ||
      owner->adapter == nullptr) {
    return;
  }
  std::unique_lock terminal_lock{owner->adapter->residency_terminal_gate};
  std::unique_lock lock{owner->gate};
  if (!owner->active || owner->final_sent || epoch >= owner->commands.size()) {
    return;
  }
  MetalResidencyScheduleCommand &entry = owner->commands[epoch];
  if (!entry.signaled || entry.completed) {
    return;
  }
  const bool known = terminal == NativeTerminal::Known;
  if (known) {
    [entry.allocator reset];
    if (owner->native_inflight == 0u ||
        !EndMetalResidencyCommand(*owner->adapter)) {
      owner->unknown = true;
    } else {
      --owner->native_inflight;
    }
  } else {
    owner->unknown = true;
    owner->adapter->residency_quarantined.store(true,
                                                std::memory_order_release);
    SetMetalLastError(*owner->adapter, "compute_device_lost");
    owner->quarantine = owner;
  }

  KernelResult result{
      .check = {true, "ok"},
      .stats =
          rund::RuntimeStats{
              .run =
                  {
                      .work =
                          {
                              .dispatch_count = entry.dispatch_count,
                              .command_submit_count = 1u,
                              .command_inflight_peak =
                                  owner->native_inflight_peak,
                              .reset_command_count = entry.reset_count,
                              .reset_bytes = entry.reset_bytes,
                          },
                  },
              .outcome = {.ok = true, .reason = "ok"},
          },
      .pipeline = {.control_command_count = entry.control_count,
                   .submitted = true},
      .terminal = owner->unknown ? NativeTerminal::UnknownMayWrite : terminal,
  };
  if (owner->unknown || entry.force_device_lost) {
    result.check = {false, "compute_device_lost"};
  } else if (!entry.admission.ok) {
    result.check = entry.admission;
  } else if (!ObserveMetalResidencyScheduleEvidence(
                 *entry.sequence, result.pipeline) ||
             !result.pipeline.control_observed ||
             result.pipeline.control.generation != entry.control_generation) {
    result.check = {false, "accel_metal_buffer_unavailable"};
  }
  result.stats.outcome.ok = result.check.ok;
  result.stats.outcome.reason = result.check.reason;
  if (result.check.ok) {
    RecordMetalDispatches(*owner->adapter, entry.dispatch_count);
  } else {
    result.stats.run.work.dispatch_count = 0u;
    result.stats.run.work.reset_command_count = 0u;
    result.stats.run.work.reset_bytes = 0u;
  }
  entry.result = result;
  entry.completed = true;

  // Emergency abort drains each bank in its natural e -> e+2 order. Opening
  // every pending gate at once would let one shared guard be overwritten by a
  // later role before the predecessor consumes it.
  if (owner->aborting && known && epoch + 2u < owner->commands.size()) {
    MetalResidencyScheduleCommand &next = owner->commands[epoch + 2u];
    static_cast<void>(signal_locked(owner, next, owner->abort_failure));
  }

  if (result.check.ok && owner->first_failure.ok) {
    ++owner->success_prefix;
  } else if (!result.check.ok && owner->first_failure.ok) {
    owner->first_failure = result.check;
  } else if (!entry.admission.ok && known) {
    if (owner->suppressed_count == 0u) {
      owner->suppressed_first = epoch;
    }
    ++owner->suppressed_count;
  }

  while (owner->release_count < owner->commands.size() &&
         owner->commands[owner->release_count].completed) {
    MetalResidencyScheduleCommand &ready =
        owner->commands[owner->release_count];
    const bool ready_unknown =
        ready.result.terminal == NativeTerminal::UnknownMayWrite;
    ready.delivery.owner = owner;
    ready.delivery.callbacks = owner->request;
    ready.delivery.release = BackendResidencyScheduleRelease{
        .receipt =
            BackendResidencyWindowReceipt{
                .check = ready.result.check,
                .terminal = ready.result.terminal,
                .epoch = ready.epoch,
                .backend_sequence = ready.epoch + 1u,
                .bank = static_cast<std::uint8_t>(ready.epoch % 2u),
                .dispatched = true,
                .completed = !ready_unknown,
                .may_write = ready_unknown || ready.admission.ok,
            },
        .result = ready.result,
    };
    ++owner->release_count;
    const bool emit_final =
        ready_unknown || owner->release_count == owner->commands.size();
    if (emit_final) {
      owner->final_sent = true;
      owner->active = false;
      ready.delivery.final = BackendResidencyScheduleFinal{
          .check = owner->first_failure,
          .terminal = ready_unknown ? NativeTerminal::UnknownMayWrite
                                    : NativeTerminal::Known,
          .plan_identity = owner->request.plan_identity,
          .token = owner->request.token,
          .generation = owner->request.generation,
          .epoch_count = owner->request.epoch_count,
          .public_handoffs = 1u,
          .native_batches = owner->request.epoch_count,
          .queue_calls = owner->request.epoch_count,
          .native_inflight_peak =
              std::max<std::uint64_t>(1u, owner->native_inflight_peak),
          .released_prefix = owner->release_count,
          .completed_prefix = owner->success_prefix,
          .suppressed_first = owner->suppressed_first,
          .suppressed_count = owner->suppressed_count,
          .completed_ns = MonotonicNanoseconds(),
      };
      ready.delivery.emit_final = true;
      clear_active_sequences(*owner);
      owner->request.release = nullptr;
      owner->request.final = nullptr;
      owner->request.user = nullptr;
    }
    dispatch_async_f(owner->service, &ready.delivery, deliver);
    if (emit_final) {
      break;
    }
  }
  terminal_lock.unlock();
  lock.unlock();
  RecordMetalCommandSubmitWaitNs(
      *owner->adapter,
      known ? MonotonicNanoseconds() - owner->submit_begin_ns : 0u);
}

#endif

#endif

} // namespace rund::node::accel::detail::schedule_detail
