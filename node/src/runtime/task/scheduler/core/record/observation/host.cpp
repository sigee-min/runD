#include "local.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool
SameStableHostEventFields(const ::rund::host::Event &expected,
                          const ::rund::host::Event &actual) noexcept {
  return expected.sequence == actual.sequence && expected.kind == actual.kind &&
         expected.status == actual.status &&
         expected.task_id == actual.task_id &&
         expected.logical_time_ns == actual.logical_time_ns &&
         expected.stream_id == actual.stream_id &&
         expected.draw_id == actual.draw_id &&
         expected.host_handle_id == actual.host_handle_id &&
         expected.offset == actual.offset &&
         expected.requested_bytes == actual.requested_bytes &&
         expected.completed_bytes == actual.completed_bytes &&
         expected.native_errno == actual.native_errno &&
         expected.name_hash.value == actual.name_hash.value &&
         expected.path_hash.value == actual.path_hash.value &&
         expected.payload_hash.value == actual.payload_hash.value;
}

} // namespace

Scheduler::HostEventCommitResult Scheduler::CommitHostEvent(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource *const source) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (state_->plan.failure != ReasonCode::Ok) {
    return HostEventCommitResult::unpublished_failure(state_->plan.failure);
  }
  const std::uint64_t physical_sequence =
      state_->identity.next_host_event_sequence++;
  if (event.task_id == 0u) {
    event.task_id = CurrentTaskId();
  }
  const std::uint64_t physical_task_id = event.task_id;
  const std::uint64_t physical_handle = event.host_handle_id;
  const std::uint64_t accepted_handle =
      event.kind == ::rund::host::EventKind::NetAccept ? event.offset : 0u;
  event.sequence = state_->plan.event(physical_sequence);
  event.task_id = state_->plan.task(physical_task_id);
  event.host_handle_id = state_->plan.handle(physical_handle);
  if (event.kind == ::rund::host::EventKind::NetAccept) {
    event.offset = state_->plan.handle(accepted_handle);
  }
  if (event.kind == ::rund::host::EventKind::IoClose &&
      event.status == ::rund::host::Status::Ok) {
    state_->plan.retire(physical_handle);
  }
  if (state_->plan.failure != ReasonCode::Ok) {
    return HostEventCommitResult::unpublished_failure(state_->plan.failure);
  }
  const std::uint64_t committed_sequence = event.sequence;
  if (source != nullptr && event.status == ::rund::host::Status::Ok &&
      event.completed_bytes == source->byte_count) {
    if (state_->evidence.input_capture_active.load(std::memory_order_relaxed) &&
        state_->evidence.host_payload_store.CapturesIngress()) {
      event.payload_hash = state_->evidence.host_payload_store.CaptureIngress(
          committed_sequence, event.kind, *source);
    } else if (event.payload_hash.value == 0u) {
      event.payload_hash = replay_detail::payload::HashIngress(*source);
    }
  }
  bool retained = false;
  if (state_->evidence.host_events.size() <
      state_->resources.limits.host_event_capacity) {
    state_->evidence.host_events.push_back(event);
    retained = true;
  } else {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::HostEventsDropped);
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::HostEvents);
  record_detail::RecordNetworkStats(state_->evidence.metrics, event);
  HashHost(state_->evidence.metrics, ::rund::host::hash_event(event).value);
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay &&
      !state_->identity.host_replay_failed) {
    const std::size_t index = state_->identity.next_expected_host_event++;
    const auto &expected_events = state_->plan.value.expected->events();
    const bool matched =
        index < expected_events.size() &&
        SameStableHostEventFields(expected_events[index], event);
    if (!matched) {
      state_->identity.host_replay_failed = true;
      state_->identity.host_replay_reason =
          ReasonString(ReasonCode::HostReplayEventMismatch);
      const std::uint64_t failed_task_id = physical_task_id;
      if (failed_task_id != 0u) {
        if (TaskRecord *const record = state_->Find(failed_task_id);
            record != nullptr) {
          const TaskState previous_state = record->state;
          record->state = TaskState::Failed;
          record->failure_code = ReasonCode::HostReplayEventMismatch;
          record->lane_segment_side_exit = true;
          if (previous_state != TaskState::Running &&
              previous_state != TaskState::Completed &&
              previous_state != TaskState::Failed) {
            ++::rund::detail::task::Stat(
                state_->evidence.metrics,
                ::rund::detail::task::StatSlot::Failed);
            RecordTerminalBatch(::rund::detail::task::OperationKind::Fail,
                                ReasonCode::HostReplayEventMismatch,
                                record->id);
            WakeJoinWaiters(record->id, ReasonCode::HostReplayEventMismatch);
            DestroyTask(*record);
          }
        }
      }
    }
  }
  const ReasonCode code = !state_->identity.host_replay_failed &&
                                  state_->plan.failure == ReasonCode::Ok
                              ? ReasonCode::Ok
                          : state_->plan.failure != ReasonCode::Ok
                              ? state_->plan.failure
                              : ReasonCode::HostReplayEventMismatch;
  return HostEventCommitResult::published(code, committed_sequence, retained);
}

} // namespace rund::node
