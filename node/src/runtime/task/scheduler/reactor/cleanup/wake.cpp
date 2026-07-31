#include "operations.hpp"

#include "../../../../reactor/readiness/handle.hpp"
#include "../../../../reactor/readiness/mask.hpp"

namespace rund::node::reactor_cancel_cleanup {

void WakeTask(Scheduler &scheduler, TaskRecord &record,
              const std::uint64_t wait_id, const ReactorHandle handle,
              const ReactorInterest interest, const ReactorEvent events,
              const std::int64_t deadline_ns, const ReasonCode reason,
              const bool cleanup_ok) noexcept {
  record.io_revents = ReactorEventBits(events);
  record.io_result = cleanup_ok ? reason : ReasonCode::IoPollFailed;
  record.state = TaskState::Ready;
  record.lane_segment_side_exit = true;
  const ReasonCode wake_code = record.io_result;
  scheduler.state_->EnqueueProgress(record);
  scheduler.Record(::rund::detail::task::OperationKind::IoWake, wake_code,
                   record.id, 0u, wait_id, 0u, ReactorHandleForPublic(handle),
                   ReactorInterestBits(interest), ReactorEventBits(events),
                   deadline_ns);
}

} // namespace rund::node::reactor_cancel_cleanup
