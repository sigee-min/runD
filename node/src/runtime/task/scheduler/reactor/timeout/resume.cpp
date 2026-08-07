#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

#include "../cleanup/request.hpp"

namespace rund::node {

::rund::detail::task::IoDecision
Scheduler::ResumeTimedReactorWait(TaskRecord &record,
                                  const std::uint64_t wait_id) noexcept {
  if (!record.coroutine_task) {
    record.wait_id = 0u;
    SetLeafFailure(record, ReasonCode::TaskLeafPrimitiveForbidden);
    static_cast<void>(ReactorCleanupWait(
        *this, ReactorCleanupRequest{.wait_id = wait_id,
                                     .reason = ReasonCode::IoPollFailed,
                                     .cancel_timeout_timer = true,
                                     .require_timeout_timer_cancel = true,
                                     .remove_ready_backlog = true}));
    ::rund::detail::task::IoDecision result =
        FailIo(ReasonCode::TaskLeafPrimitiveForbidden);
    CompletePrimitiveCommit();
    return result;
  }
  record.dynamic_scope_id = CurrentScopeId();
  record.lane_segment_side_exit = true;
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::CoroutineParks),
      1u);
  record.coroutine_parked = true;
  return ::rund::detail::task::IoDecision{.status = task::Status::success(),
                                          .suspend = true};
}

} // namespace rund::node
