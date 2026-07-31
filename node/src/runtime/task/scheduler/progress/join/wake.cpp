#include "../../state/model/join.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

void Scheduler::WakeJoinWaiters(const std::uint64_t target_task_id,
                                const ReasonCode result) noexcept {
  std::vector<JoinWait> &waits = state_->ready.join_waits;
  std::size_t write = 0u;
  for (std::size_t read = 0u; read < waits.size(); ++read) {
    const JoinWait wait = waits[read];
    if (wait.target_task_id != target_task_id) {
      if (write != read) {
        waits[write] = wait;
      }
      ++write;
      continue;
    }
    TaskRecord *const waiter = state_->Find(wait.waiter_task_id);
    if (waiter == nullptr || waiter->state != TaskState::JoinBlocked ||
        waiter->wait_id != wait.wait_id ||
        waiter->wait_source_id != target_task_id) {
      continue;
    }
    waiter->wait_result = result;
    waiter->state = TaskState::Ready;
    waiter->lane_segment_side_exit = true;
    state_->EnqueueProgress(*waiter);
    {
      std::lock_guard lock{state_->batches.direct_mutex};
      ++state_->batches.ready_epoch;
    }
    state_->batches.direct_cv.notify_all();
    Record(::rund::detail::task::OperationKind::JoinWake, result, waiter->id,
           target_task_id, wait.wait_id);
  }
  waits.resize(write);
}

} // namespace rund::node
