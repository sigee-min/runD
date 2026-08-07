#include "resume/local.hpp"

#include "../ready/set/identity.hpp"

namespace rund::node {

::rund::net::ready::many::Wait
ReadyManyAccess::Resume(Scheduler &scheduler, ReadyManyEntry &entry,
                        const std::span<::rund::net::ready::Event> out,
                        const std::uint64_t group_id) noexcept {
  SchedulerState &state = *scheduler.state_;
  scheduler.EnsureCurrentCommit();
  TaskRecord *record = state.Find(entry.task_id);
  if (record == nullptr || record->state != TaskState::Running) {
    ::rund::net::ready::many::Wait result =
        FailManyCode(ReasonCode::TaskContextMissing);
    scheduler.CompletePrimitiveCommit();
    return result;
  }
  const ReasonCode wake_code = record->io_result;
  const bool timed_out = wake_code == ReasonCode::IoTimedOut;
  if (wake_code == ReasonCode::TaskCancelled) {
    ResetReadyManyResumeTask(*record);
    ::rund::net::ready::many::Result result = MakeReadyManyCancelledResult();
    scheduler.CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }

  ReactorManyGroup *group =
      ReactorManyFindGroup(state.reactor.reactor_many_groups, group_id);
  if (group == nullptr) {
    ResetReadyManyResumeWaitState(*record);
    ::rund::net::ready::many::Wait result =
        FailManyCode(ReasonCode::TaskContextMissing);
    scheduler.CompletePrimitiveCommit();
    return result;
  }

  ReasonCode ready_code = timed_out ? ReasonCode::IoTimedOut : wake_code;
  const bool group_budget_exhausted = group->budget_exhausted;
  const bool ready_set_wait =
      !ReactorReadySetIdentityOwner::empty(group->ready_set);
  std::uint32_t copied = 0u;
  const bool copied_ok =
      scheduler.CopyReactorManyEvents(group_id, out, &copied);
  if (!copied_ok && ready_code == ReasonCode::Ok) {
    ready_code = ReasonCode::IoPollFailed;
  }
  if (!timed_out && ready_code == ReasonCode::Ok && copied == 0u) {
    ready_code = ReasonCode::IoPollFailed;
  }
  if (ready_set_wait) {
    RecordReactorReadySetEvents(state.evidence.metrics, copied);
  }
  const bool cleanup_ok =
      CleanupReadyManyResume(scheduler, group_id, ready_code);
  ResetReadyManyResumeTask(*record);

  const ReasonCode result_code =
      ReadyManyResumeResultCode(ready_code, cleanup_ok);
  ::rund::net::ready::many::Result result =
      MakeReadyManyResumeResult(result_code, copied, group_budget_exhausted);
  scheduler.CompletePrimitiveCommit();
  return ::rund::net::ready::many::detail::Access::Complete(result);
}

} // namespace rund::node
