#include <rund/task/stats/slots.hpp>

#include "../state/model/task.hpp"
#include "../state/storage.hpp"

namespace rund::node {

::rund::detail::task::ChannelDecision
Scheduler::ParkChannel(const std::uint64_t channel_id,
                       const bool send_wait) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      send_wait ? ::rund::detail::task::OperationKind::ChannelSend
                : ::rund::detail::task::OperationKind::ChannelRecv);
  EnsureCurrentCommit();
  TaskRecord *record = state_->Find(CurrentTaskId());
  if (record == nullptr || record->state != TaskState::Running) {
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::TaskContextMissing)};
  }
  if (!record->coroutine_task) {
    SetLeafFailure(*record, ReasonCode::TaskLeafPrimitiveForbidden);
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::TaskLeafPrimitiveForbidden)};
  }
  if (state_->resources.live_channel_waits >=
      state_->resources.limits.channel_wait_capacity) {
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::ChannelWaitCapacityExceeded)};
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  ++state_->resources.live_channel_waits;
  const std::uint64_t wait_id = state_->identity.next_wait_id++;
  record->state = TaskState::ChannelBlocked;
  record->wait_id = wait_id;
  record->wait_source_id = channel_id;
  record->wait_result = ReasonCode::Ok;
  record->dynamic_scope_id = CurrentScopeId();
  record->coroutine_parked = true;
  record->lane_segment_side_exit = true;
  const ::rund::detail::task::OperationKind park_kind =
      send_wait ? ::rund::detail::task::OperationKind::ChannelSend
                : ::rund::detail::task::OperationKind::ChannelRecv;
  Record(park_kind, ReasonCode::Ok, record->id, 0u, wait_id, channel_id);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::CoroutineParks);
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success(),
                                               .suspend = true,
                                               .task_id = record->id};
}

::rund::detail::task::ChannelDecision
Scheduler::WakeChannel(const std::uint64_t task_id,
                       const std::uint64_t channel_id) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::ChannelWake);
  EnsureCurrentCommit();
  TaskRecord *const record = state_->Find(task_id);
  if (record == nullptr) {
    return ::rund::detail::task::ChannelDecision{
        .status = task::Status::fail(ReasonCode::TaskHandleUnknown)};
  }
  if (record->state == TaskState::Ready &&
      record->wait_source_id == channel_id &&
      record->wait_result == ReasonCode::TaskDeadlock) {
    record->wait_result = ReasonCode::Ok;
    Record(::rund::detail::task::OperationKind::ChannelWake, ReasonCode::Ok,
           record->id, 0u, record->wait_id, channel_id);
    return ::rund::detail::task::ChannelDecision{.status =
                                                     task::Status::success()};
  }
  if (record->state == TaskState::ChannelBlocked) {
    const std::uint64_t wait_id = record->wait_id;
    record->state = TaskState::Ready;
    record->lane_segment_side_exit = true;
    record->wait_result = ReasonCode::Ok;
    if (state_->resources.live_channel_waits > 0u) {
      --state_->resources.live_channel_waits;
    }
    state_->EnqueueProgress(*record);
    Record(::rund::detail::task::OperationKind::ChannelWake, ReasonCode::Ok,
           record->id, 0u, wait_id, channel_id);
  }
  return ::rund::detail::task::ChannelDecision{.status =
                                                   task::Status::success()};
}

} // namespace rund::node
