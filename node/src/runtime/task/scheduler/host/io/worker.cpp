#include "local.hpp"
#include "../../state/model/task.hpp"

#include <cerrno>

namespace rund::node::host_io {
namespace {

[[nodiscard]] HostIoOutcome Outcome(const NativeIoResult native) noexcept {
  return HostIoOutcome{
      .value = native.value,
      .error = native.err,
      .kind = native.unsupported
                  ? HostIoOutcomeKind::Unsupported
                  : (native.invalid_buffer ? HostIoOutcomeKind::InvalidBuffer
                                           : HostIoOutcomeKind::Ready),
  };
}

void FailWake(SchedulerState &scheduler_state, HostIoSlot &slot) noexcept {
  {
    std::lock_guard lock{scheduler_state.evidence.mutex};
    TaskRecord *const record = scheduler_state.Find(slot.wake.id);
    if (record != nullptr && record == slot.wake.record &&
        record->state == TaskState::ExternalBlocked) {
      record->state = TaskState::Failed;
      record->failure_code = ReasonCode::TaskStateTransitionInvalid;
      record->coroutine_parked = false;
    }
  }
  {
    std::lock_guard lock{scheduler_state.batches.direct_mutex};
    if (scheduler_state.batches.direct_jobs_in_flight != 0u) {
      --scheduler_state.batches.direct_jobs_in_flight;
    }
    if (scheduler_state.batches.task_direct_jobs_in_flight != 0u) {
      --scheduler_state.batches.task_direct_jobs_in_flight;
    }
  }
  scheduler_state.batches.direct_cv.notify_all();
  Release(scheduler_state.host_io, slot);
}

} // namespace

void Run(Scheduler &scheduler, SchedulerState &scheduler_state) {
  SchedulerHostIoState &state = scheduler_state.host_io;
  for (;;) {
    HostIoSlot *slot = nullptr;
    {
      std::unique_lock lock{state.mutex};
      state.ready.wait(lock, [&state] {
        return state.queue_head != nullptr ||
               (state.stop && state.submissions_in_progress == 0u);
      });
      if (state.queue_head == nullptr) {
        if (state.stop && state.submissions_in_progress == 0u) {
          return;
        }
        continue;
      }
      slot = state.queue_head;
      state.queue_head = slot->next;
      if (state.queue_head == nullptr) {
        state.queue_tail = nullptr;
      }
      slot->next = nullptr;
      slot->state.store(HostIoSlotState::Running, std::memory_order_release);
      if (slot->operation.sequence != state.next_execution_sequence) {
        slot->outcome = HostIoOutcome{
            .value = -1,
            .error = EINVAL,
            .kind = HostIoOutcomeKind::InvalidBuffer,
        };
      } else {
        ++state.next_execution_sequence;
      }
    }

    if (slot->outcome.kind == HostIoOutcomeKind::Pending) {
      const HostIoOperation &operation = slot->operation;
      slot->outcome =
          operation.kind == HostIoKind::Read
              ? Outcome(NativeRead(operation.native, operation.read_buffer()))
              : Outcome(
                    NativeWrite(operation.native, operation.write_buffer()));
    }
    slot->state.store(HostIoSlotState::Complete, std::memory_order_release);

    const std::uint64_t sequence = slot->operation.sequence;
    const std::uint8_t phase =
        slot->phase.exchange(3u, std::memory_order_acq_rel);
    if (phase != 2u || !scheduler.WakeExternal(slot->wake)) {
      FailWake(scheduler_state, *slot);
      continue;
    }

    std::unique_lock lock{state.mutex};
    state.ready.wait(
        lock, [slot, sequence] { return slot->released_sequence == sequence; });
  }
}

} // namespace rund::node::host_io
