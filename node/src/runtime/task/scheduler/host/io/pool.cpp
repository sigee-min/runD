#include "local.hpp"

#include <limits>
#include <new>

namespace rund::node {

SchedulerHostIoState::SchedulerHostIoState() noexcept = default;
SchedulerHostIoState::~SchedulerHostIoState() = default;

namespace host_io {
namespace {

void ResetLocked(SchedulerHostIoState &state, HostIoSlot &slot) noexcept {
  if (slot.state.load(std::memory_order_relaxed) ==
          HostIoSlotState::Admitting &&
      state.submissions_in_progress != 0u) {
    --state.submissions_in_progress;
  }
  slot.phase.store(0u, std::memory_order_release);
  slot.wake = {};
  slot.operation = {};
  slot.outcome = {};
  slot.state.store(HostIoSlotState::Free, std::memory_order_release);
  slot.next = state.free_head;
  state.free_head = &slot;
}

} // namespace

HostIoSlot *Claim(Scheduler &scheduler, SchedulerState &scheduler_state,
                  const HostIoOperation &operation) noexcept {
  SchedulerHostIoState &state = scheduler_state.host_io;
  std::lock_guard lock{state.mutex};
  if (state.stop || state.slots == nullptr) {
    return nullptr;
  }
  if (!state.worker.joinable()) {
    try {
      state.worker = std::thread{
          [&scheduler, &scheduler_state] { Run(scheduler, scheduler_state); }};
    } catch (...) {
      return nullptr;
    }
  }
  HostIoSlot *const slot = state.free_head;
  if (slot == nullptr || state.next_submission_sequence ==
                             std::numeric_limits<std::uint64_t>::max()) {
    return nullptr;
  }
  state.free_head = slot->next;
  slot->operation = operation;
  slot->operation.sequence = state.next_submission_sequence++;
  slot->next = nullptr;
  slot->outcome = {};
  slot->state.store(HostIoSlotState::Admitting, std::memory_order_release);
  ++state.submissions_in_progress;
  return slot;
}

void Queue(SchedulerHostIoState &state, HostIoSlot &slot) noexcept {
  {
    std::lock_guard lock{state.mutex};
    if (slot.state.load(std::memory_order_acquire) !=
        HostIoSlotState::Admitting) {
      return;
    }
    if (state.submissions_in_progress != 0u) {
      --state.submissions_in_progress;
    }
    slot.state.store(HostIoSlotState::Queued, std::memory_order_release);
    slot.next = nullptr;
    if (state.queue_tail == nullptr) {
      state.queue_head = &slot;
    } else {
      state.queue_tail->next = &slot;
    }
    state.queue_tail = &slot;
  }
  state.ready.notify_one();
}

void Release(SchedulerHostIoState &state, HostIoSlot &slot) noexcept {
  {
    std::lock_guard lock{state.mutex};
    const std::uint64_t sequence = slot.operation.sequence;
    ResetLocked(state, slot);
    slot.released_sequence = sequence;
  }
  state.ready.notify_all();
}

bool Valid(SchedulerHostIoState &state, const HostIoSlot *const slot,
           const HostIoKind kind) noexcept {
  std::lock_guard lock{state.mutex};
  return slot != nullptr && slot->operation.kind == kind &&
         slot->state.load(std::memory_order_acquire) ==
             HostIoSlotState::Complete;
}

} // namespace host_io

bool Scheduler::PrepareHostIo() noexcept {
  SchedulerHostIoState &state = state_->host_io;
  const std::size_t capacity =
      static_cast<std::size_t>(state_->resources.limits.host_io_capacity);
  if (capacity == state.capacity &&
      (capacity == 0u || state.slots != nullptr)) {
    return true;
  }
  if (capacity == 0u) {
    state.slots.reset();
    state.capacity = 0u;
    state.free_head = nullptr;
    return true;
  }

  try {
    auto slots = std::make_unique<HostIoSlot[]>(capacity);
    HostIoSlot *free = nullptr;
    for (std::size_t index = capacity; index != 0u; --index) {
      slots[index - 1u].next = free;
      free = &slots[index - 1u];
    }
    state.slots = std::move(slots);
    state.capacity = capacity;
    state.free_head = free;
    return true;
  } catch (...) {
    return false;
  }
}

void Scheduler::StopHostIo() noexcept {
  SchedulerHostIoState &state = state_->host_io;
  {
    std::lock_guard lock{state.mutex};
    state.stop = true;
  }
  state.ready.notify_all();
  if (state.worker.joinable()) {
    state.worker.join();
  }

  std::lock_guard lock{state.mutex};
  state.queue_head = nullptr;
  state.queue_tail = nullptr;
  state.submissions_in_progress = 0u;
  state.stop = false;
}

} // namespace rund::node
