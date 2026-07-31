#include "../../state/model/lane.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

void Scheduler::QueueDirect(TaskLane& lane, TaskRecord& record,
                            const std::uint64_t ticket,
                            const std::uint64_t sequence) noexcept {
  {
    std::lock_guard direct_lock{state_->batches.direct_mutex};
    ++state_->batches.direct_jobs_in_flight;
    ++state_->batches.task_direct_jobs_in_flight;
  }
  const bool busy =
      lane.running || lane.has_job || lane.nested_job_active ||
      lane.root_reserved.load(std::memory_order_acquire) ||
      lane.completed_job_sequence.load(std::memory_order_acquire) != 0u ||
      lane.mailbox_state.load(std::memory_order_acquire) != 0u;
  if (busy) {
    record.wake_ticket = ticket;
    record.wait_token = sequence;
    record.wake_next = nullptr;
    if (lane.direct_ready_tail == nullptr) {
      lane.direct_ready_head = &record;
    } else {
      lane.direct_ready_tail->wake_next = &record;
    }
    lane.direct_ready_tail = &record;
    return;
  }
  lane.mailbox_task_id.store(record.id, std::memory_order_relaxed);
  lane.mailbox_record.store(&record, std::memory_order_relaxed);
  lane.mailbox_commit_ticket.store(ticket, std::memory_order_relaxed);
  lane.mailbox_job_sequence.store(sequence, std::memory_order_relaxed);
  lane.mailbox_state.store(1u, std::memory_order_release);
}

} // namespace rund::node
