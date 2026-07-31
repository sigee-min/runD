#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

ReasonCode
Scheduler::ValidateSpawnBudget(const ReadyAdmission admission) noexcept {
  if (state_->resources.limits.task_capacity == 0u ||
      (state_->ready.records.size() >= state_->resources.limits.task_capacity &&
       state_->ready.free_record_head == 0u)) {
    return ReasonCode::TaskCapacityExceeded;
  }
  if (admission == ReadyAdmission::Spawn &&
      state_->ready.ready_depth >=
      state_->resources.limits.ready_queue_capacity) {
    return ReasonCode::ReadyQueueCapacityExceeded;
  }
  return ReasonCode::Ok;
}

}  // namespace rund::node
