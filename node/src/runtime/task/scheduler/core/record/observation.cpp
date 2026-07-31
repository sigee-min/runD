#include <rund/task/stats/slots.hpp>

#include "../../../../replay/host/payload/hash.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../../state/storage/check.hpp"
#include <rund/counter.hpp>

#include <limits>

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

void RecordNetworkStats(::rund::detail::task::StatStorage &stats,
                        const ::rund::host::Event &event) noexcept {
  if (event.status == ::rund::host::Status::WouldBlock) {
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkWouldBlock);
  }
  if (event.status != ::rund::host::Status::Ok) {
    return;
  }
  ::rund::detail::task::StatSlot byte_slot =
      ::rund::detail::task::StatSlot::Count;
  switch (event.kind) {
  case ::rund::host::EventKind::NetSocket:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketsOpened);
    break;
  case ::rund::host::EventKind::NetBind:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketsBound);
    break;
  case ::rund::host::EventKind::NetListen:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketsListened);
    break;
  case ::rund::host::EventKind::NetShutdown:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketsShutdown);
    break;
  case ::rund::host::EventKind::NetLocalAddress:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkLocalAddressReads);
    break;
  case ::rund::host::EventKind::NetAccept:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkAccepts);
    break;
  case ::rund::host::EventKind::NetConnect:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkConnects);
    break;
  case ::rund::host::EventKind::NetRecv:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkRecvCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSend:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSendCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  case ::rund::host::EventKind::NetRecvDatagram:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkDatagramRecvCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSendDatagram:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkDatagramSendCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  case ::rund::host::EventKind::NetSetSocketOption:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketOptionsSet);
    break;
  case ::rund::host::EventKind::NetGetSocketOption:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkSocketOptionsRead);
    break;
  case ::rund::host::EventKind::NetRecvVectored:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkVectoredRecvCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesReceived;
    break;
  case ::rund::host::EventKind::NetSendVectored:
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::NetworkVectoredSendCalls);
    byte_slot = ::rund::detail::task::StatSlot::NetworkBytesSent;
    break;
  default:
    break;
  }
  if (byte_slot != ::rund::detail::task::StatSlot::Count) {
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(stats, byte_slot), event.completed_bytes);
  }
}

[[nodiscard]] bool FitsHostPayloadCapacity(const SchedulerState &state,
                                           const std::size_t bytes) noexcept {
  const std::uint64_t current =
      state.evidence.host_payload_store.logical_bytes();
  const std::uint64_t reserved = state.evidence.host_payload_reserved_bytes;
  const std::uint64_t limit =
      state.resources.limits.host_payload_capacity_bytes;
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (current > max - reserved) {
    return false;
  }
  const std::uint64_t used = current + reserved;
  const std::uint64_t requested = static_cast<std::uint64_t>(bytes);
  if (requested > max - used) {
    return false;
  }
  return used + requested <= limit;
}

} // namespace

void Scheduler::RecordObservation(const task::ObservationKind kind,
                                  const ReasonCode code,
                                  const std::uint64_t task_id,
                                  const std::uint64_t wait_id, const int fd,
                                  const short interest, const short revents,
                                  const std::int64_t deadline_ns) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  const std::uint64_t physical_sequence =
      state_->identity.next_observation_sequence++;
  task::Observation observation{
      .sequence = state_->plan.observation(physical_sequence),
      .kind = kind,
      .task_id = state_->plan.task(task_id),
      .wait_id = state_->plan.wait(wait_id),
      .fd = state_->plan.descriptor(fd),
      .interest = interest,
      .revents = revents,
      .deadline_ns = deadline_ns,
      .reason_code = code,
  };
  if (state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Observations);
  if (state_->resources.limits.observation_capacity == 0u ||
      state_->evidence.observations.size() <
          state_->resources.limits.observation_capacity) {
    state_->evidence.observations.push_back(observation);
  } else {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::ObservationDropped);
  }
  HashObservation(state_->evidence.metrics, observation);
}

Scheduler::HostEventCommitResult Scheduler::CommitHostEvent(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource *const source) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  HostEventCommitResult result{};
  if (state_->plan.failure != ReasonCode::Ok) {
    result.code = state_->plan.failure;
    result.reason = ReasonString(result.code);
    return result;
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
    result.code = state_->plan.failure;
    result.reason = ReasonString(result.code);
    return result;
  }
  result.sequence = event.sequence;
  if (source != nullptr && event.status == ::rund::host::Status::Ok &&
      event.completed_bytes == source->byte_count) {
    if (state_->evidence.input_capture_active.load(std::memory_order_relaxed) &&
        state_->evidence.host_payload_store.CapturesIngress()) {
      event.payload_hash = state_->evidence.host_payload_store.CaptureIngress(
          result.sequence, event.kind, *source);
    } else if (event.payload_hash.value == 0u) {
      event.payload_hash = replay_detail::payload::HashIngress(*source);
    }
  }
  if (state_->evidence.host_events.size() <
      state_->resources.limits.host_event_capacity) {
    state_->evidence.host_events.push_back(event);
    result.retained = true;
  } else {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::HostEventsDropped);
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::HostEvents);
  RecordNetworkStats(state_->evidence.metrics, event);
  HashHost(state_->evidence.metrics, ::rund::host::hash_event(event).value);
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay &&
      !state_->identity.host_replay_failed) {
    const std::size_t index = state_->identity.next_expected_host_event++;
    const auto &expected_events = state_->plan.value.expected->events();
    result.expected_index = index;
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
  result.ok = !state_->identity.host_replay_failed &&
              state_->plan.failure == ReasonCode::Ok;
  result.code = result.ok ? ReasonCode::Ok
                : state_->plan.failure != ReasonCode::Ok
                    ? state_->plan.failure
                    : ReasonCode::HostReplayEventMismatch;
  result.reason = result.ok ? "ok" : ReasonString(result.code);
  return result;
}

bool Scheduler::RecordHostEvent(::rund::host::Event event) noexcept {
  return CommitHostEvent(event).ok;
}

bool Scheduler::CapturesNetIngress() const noexcept {
  return state_->evidence.input_capture_active.load(
             std::memory_order_acquire) &&
         state_->evidence.host_payload_store.CapturesIngress();
}

bool Scheduler::ReserveHostPayloadCapacity(const std::size_t bytes) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (!FitsHostPayloadCapacity(*state_, bytes)) {
    return false;
  }
  state_->evidence.host_payload_reserved_bytes +=
      static_cast<std::uint64_t>(bytes);
  return true;
}

void Scheduler::ReleaseHostPayloadCapacity(const std::size_t bytes) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  const std::uint64_t reserved = static_cast<std::uint64_t>(bytes);
  if (reserved > state_->evidence.host_payload_reserved_bytes) {
    state_->evidence.host_payload_reserved_bytes = 0u;
    return;
  }
  state_->evidence.host_payload_reserved_bytes -= reserved;
}

void Scheduler::FailCurrentTaskOrScheduler(const ReasonCode code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const char *const reason = ReasonString(code);
  const std::uint64_t task_id = CurrentTaskId();
  if (task_id != 0u) {
    if (TaskRecord *const record = state_->Find(task_id); record != nullptr) {
      const TaskState previous_state = record->state;
      record->state = TaskState::Failed;
      record->failure_code = code;
      record->lane_segment_side_exit = true;
      if (previous_state != TaskState::Running &&
          previous_state != TaskState::Completed &&
          previous_state != TaskState::Failed) {
        ++::rund::detail::task::Stat(state_->evidence.metrics,
                                     ::rund::detail::task::StatSlot::Failed);
        RecordTerminalBatch(::rund::detail::task::OperationKind::Fail, code,
                            record->id);
        WakeJoinWaiters(record->id, code);
        DestroyTask(*record);
      }
    }
    return;
  }
  state_->identity.host_replay_failed = true;
  state_->identity.host_replay_reason = reason;
  state_->identity.host_replay_payload_failed = true;
  state_->identity.host_replay_payload_reason = reason;
}

ReasonCode Scheduler::RecordHostPayloadForCommittedEvent(
    const HostEventCommitResult &commit, const ::rund::host::EventKind kind,
    const replay_detail::payload::Capture &payload) noexcept {
  ReasonCode failure = ReasonCode::Ok;
  {
    std::lock_guard lock{state_->evidence.mutex};
    state_->RequireSequencer();
    if (!commit.ok) {
      return commit.code;
    }
    if (!commit.retained) {
      return ReasonCode::Ok;
    }
    if (!payload || !FitsHostPayloadCapacity(*state_, payload.bytes().size())) {
      failure = ReasonCode::TaskCapacityExceeded;
    } else {
      try {
        if (!state_->evidence.host_payload_store.Append(commit.sequence, kind,
                                                        payload)) {
          failure = ReasonCode::HostReplayPayloadMismatch;
        }
      } catch (...) {
        failure = ReasonCode::HostReplayPayloadMismatch;
      }
    }
  }
  if (failure != ReasonCode::Ok) {
    FailCurrentTaskOrScheduler(failure);
  }
  return failure;
}

bool Scheduler::RecordHostEvents(
    const std::vector<::rund::host::Event> &events) noexcept {
  bool ok = true;
  for (const ::rund::host::Event &event : events) {
    ok = RecordHostEvent(event) && ok;
  }
  return ok;
}

bool Scheduler::RecordHostEventFromHostApi(::rund::host::Event event) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const bool recorded = RecordHostEvent(event);
  CompletePrimitiveCommit();
  return recorded;
}

bool Scheduler::RecordHostEventFromHostApi(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource &source) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const HostEventCommitResult committed = CommitHostEvent(event, &source);
  CompletePrimitiveCommit();
  return committed.ok;
}

} // namespace rund::node
