#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

namespace rund::node {

::rund::detail::task::IoDecision
Scheduler::ResumeTimedReactorWait(TaskRecord &record) noexcept {
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
