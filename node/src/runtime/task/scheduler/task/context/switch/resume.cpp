#include <rund/task/stats/slots.hpp>

#include "../../../state/model/task.hpp"
#include "../../../state/storage.hpp"

#include <rund/task/coroutine.hpp>

namespace rund::node {

bool Scheduler::ResumeCoroutine(TaskRecord &record) noexcept {
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::CoroutineResumes);
  const CompletionLease completion =
      state_->resources.completion_pool.lease(record.completion);
  if (completion) {
    task::Phase phase = CompletionPool::poll(completion).phase;
    if (phase == task::Phase::Parked) {
      (void)CompletionPool::transition(completion, task::Phase::Ready);
      phase = task::Phase::Ready;
    }
    if (phase != task::Phase::Ready ||
        !CompletionPool::transition(completion, task::Phase::Running)) {
      record.state = TaskState::Failed;
      record.failure_code = ReasonCode::TaskStateTransitionInvalid;
    }
  }
  try {
    if (record.state != TaskState::Failed &&
        record.coroutine_frame != nullptr) {
      record.coroutine_frame.resume();
    } else {
      record.state = TaskState::Failed;
      record.failure_code = ReasonCode::TaskInvalid;
    }
  } catch (...) {
    record.state = TaskState::Failed;
    record.failure_code = ReasonCode::TaskFailed;
  }
  if (record.state == TaskState::Failed) {
    return false;
  }
  if (record.coroutine_frame == nullptr || !record.coroutine_frame.done()) {
    const bool parked = (record.state == TaskState::Ready ||
                         record.state == TaskState::Sleeping ||
                         record.state == TaskState::JoinBlocked ||
                         record.state == TaskState::ChannelBlocked ||
                         record.state == TaskState::IoBlocked ||
                         record.state == TaskState::ExternalBlocked) &&
                        record.coroutine_parked;
    if (parked && completion) {
      (void)CompletionPool::transition(completion, task::Phase::Parked);
    }
    if (!parked) {
      record.state = TaskState::Failed;
      record.failure_code = ReasonCode::TaskCoroutinePrimitiveAwaitNotLive;
    }
    return parked;
  }
  try {
    if (record.coroutine_ops == nullptr ||
        record.coroutine_ops->result == nullptr ||
        (completion &&
         !CompletionPool::transition(completion, task::Phase::Committing))) {
      record.state = TaskState::Failed;
      record.failure_code = ReasonCode::TaskStateTransitionInvalid;
      return false;
    }
    const ::rund::detail::task::CoroutineResult result =
        record.coroutine_ops->result(record.coroutine_frame);
    const bool result_valid =
        result.code == ReasonCode::Ok &&
        (!result.has_value ||
         (result.value != nullptr && result.type != nullptr &&
          result.move != nullptr && result.destroy != nullptr));
    const bool result_observed = completion || !result.has_value;
    record.state = result_valid && result_observed ? TaskState::Completed
                                                   : TaskState::Failed;
    record.failure_code =
        result_valid && result_observed
            ? ReasonCode::Ok
            : (result.code == ReasonCode::Ok ? ReasonCode::TaskFailed
                                             : result.code);
    if (completion) {
      const task::Status prepared = CompletionPool::prepare(completion, result);
      if (!prepared && prepared.code() != result.code) {
        record.state = TaskState::Failed;
        record.failure_code = prepared.code();
        (void)CompletionPool::prepare_failure(completion, prepared.code());
      }
    }
  } catch (...) {
    record.state = TaskState::Failed;
    record.failure_code = ReasonCode::TaskFailed;
    if (completion) {
      (void)CompletionPool::prepare_failure(completion, ReasonCode::TaskFailed);
    }
  }
  return false;
}

} // namespace rund::node
