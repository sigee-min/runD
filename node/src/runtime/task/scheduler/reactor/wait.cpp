#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../../reactor/platform.hpp"
#include "../../../reactor/readiness/handle.hpp"
#include "../../../reactor/readiness/mask.hpp"
#include "../state/model/task.hpp"
#include "../state/storage.hpp"
#include "generation.hpp"
#include "poll.hpp"
#include "registry.hpp"
#include "state.hpp"
#include "stats.hpp"

namespace rund::node {
::rund::detail::task::IoDecision
Scheduler::WaitReactor(const int fd, const short interest,
                       const std::uint64_t host_handle_id,
                       const std::uint64_t fd_generation,
                       const ::rund::net::SocketView socket) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::IoPark);
  EnsureCurrentCommit();
  const std::uint64_t task_id = CurrentTaskId();
  TaskRecord *record = state_->Find(task_id);
  if (record == nullptr || record->state != TaskState::Running) {
    ::rund::detail::task::IoDecision result =
        FailIo(ReasonCode::TaskContextMissing);
    CompletePrimitiveCommit();
    return result;
  }
  if (!record->coroutine_task) {
    SetLeafFailure(*record, ReasonCode::TaskLeafPrimitiveForbidden);
    ::rund::detail::task::IoDecision result =
        FailIo(ReasonCode::TaskLeafPrimitiveForbidden);
    CompletePrimitiveCommit();
    return result;
  }
  if (fd < 0) {
    ::rund::detail::task::IoDecision result = FailIo(ReasonCode::IoFdInvalid);
    CompletePrimitiveCommit();
    return result;
  }
  const std::uint64_t wait_host_handle_id =
      ReactorHostHandleId(fd, host_handle_id);
  if (fd_generation != 0u) {
    ReasonCode generation_failure = ReasonCode::Ok;
    if (!ReactorGenerationCleanupStaleWaits(*this, ReactorHandleFromPublic(fd),
                                            fd_generation,
                                            &generation_failure)) {
      ::rund::detail::task::IoDecision result = FailIo(generation_failure);
      CompletePrimitiveCommit();
      return result;
    }
  }
  const ReactorProbeResult immediate = ReactorProbeNow(
      state_->reactor.reactor.platform, state_->reactor.reactor.probe_ready,
      ReactorHandleFromPublic(fd), ReactorInterestFromBits(interest));
  const ReactorEvent immediate_events = immediate.events();
  const short immediate_revents = ReactorEventBits(immediate_events);
  switch (immediate.disposition()) {
  case ReactorProbeDisposition::BackendUnavailable: {
    ::rund::detail::task::IoDecision failure =
        FailIo(ReasonCode::ReactorBackendUnavailable);
    CompletePrimitiveCommit();
    return failure;
  }
  case ReactorProbeDisposition::PollFailed: {
    RecordReactorObservation(task::ObservationKind::IoPollFailed,
                             ReasonCode::IoPollFailed, record->id, 0u, fd,
                             interest, 0);
    if (!RecordReactorHostEvent(ReasonCode::IoPollFailed, record->id,
                                wait_host_handle_id)) {
      ::rund::detail::task::IoDecision failure =
          FailIo(ReasonCode::HostReplayEventMismatch);
      CompletePrimitiveCommit();
      return failure;
    }
    ::rund::detail::task::IoDecision failure = FailIo(ReasonCode::IoPollFailed);
    CompletePrimitiveCommit();
    return failure;
  }
  case ReactorProbeDisposition::Invalid: {
    RecordReactorObservation(task::ObservationKind::IoInvalid,
                             ReasonCode::IoFdInvalid, record->id, 0u, fd,
                             interest, immediate_revents);
    if (!RecordReactorHostEvent(ReasonCode::IoFdInvalid, record->id,
                                wait_host_handle_id)) {
      ::rund::detail::task::IoDecision failure =
          FailIo(ReasonCode::HostReplayEventMismatch);
      CompletePrimitiveCommit();
      return failure;
    }
    ::rund::detail::task::IoDecision failure =
        FailIo(ReasonCode::IoFdInvalid, immediate_revents);
    CompletePrimitiveCommit();
    return failure;
  }
  case ReactorProbeDisposition::Ready:
    if (ReactorEventsMatch(immediate_events,
                           ReactorInterestFromBits(interest))) {
      RecordReactorObservation(task::ObservationKind::IoReady, ReasonCode::Ok,
                               record->id, 0u, fd, interest, immediate_revents);
      if (!RecordReactorHostEvent(ReasonCode::Ok, record->id,
                                  wait_host_handle_id)) {
        ::rund::detail::task::IoDecision failure =
            FailIo(ReasonCode::HostReplayEventMismatch);
        CompletePrimitiveCommit();
        return failure;
      }
      ::rund::detail::task::IoDecision ready_result{
          .status = task::Status::success(), .revents = immediate_revents};
      CompletePrimitiveCommit();
      return ready_result;
    }
    break;
  case ReactorProbeDisposition::NotReady:
    break;
  }
  for (;;) {
    if (ReactorRegistrySize(state_->reactor.reactor) >=
        state_->resources.limits.reactor_wait_capacity) {
      ::rund::detail::task::IoDecision result =
          FailIo(ReasonCode::ReactorWaitCapacityExceeded);
      CompletePrimitiveCommit();
      return result;
    }
    const std::uint64_t wait_id = state_->identity.next_wait_id++;
    const ReactorWait wait{
        .socket = socket,
        .task_id = record->id,
        .wait_id = wait_id,
        .host_handle_id = wait_host_handle_id,
        .fd_generation = fd_generation,
        .fd = ReactorHandleFromPublic(fd),
        .interest = ReactorInterestFromBits(interest),
    };
    if (!ReactorRegistryAddWait(state_->reactor.reactor, wait)) {
      ::rund::detail::task::IoDecision result =
          FailIo(ReasonCode::ReactorWaitCapacityExceeded);
      CompletePrimitiveCommit();
      return result;
    }
    if (!ReactorRegistryCollectChangesForWaitAdd(state_->reactor.reactor,
                                                 wait)) {
      (void)ReactorRegistryRemoveWait(state_->reactor.reactor, wait.wait_id,
                                      nullptr);
      ::rund::detail::task::IoDecision result =
          FailIo(ReasonCode::ReactorWaitCapacityExceeded);
      CompletePrimitiveCommit();
      return result;
    }
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::ReactorWaits),
        1u);
    RecordReactorWaitRegistered(state_->evidence.metrics);
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Parked),
        1u);
    record->state = TaskState::IoBlocked;
    record->wait_id = wait_id;
    Record(::rund::detail::task::OperationKind::IoPark, ReasonCode::Ok,
           record->id, 0u, wait_id, 0u, fd, interest, 0);
    record->dynamic_scope_id = CurrentScopeId();
    record->lane_segment_side_exit = true;
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::CoroutineParks),
        1u);
    record->coroutine_parked = true;
    return ::rund::detail::task::IoDecision{.status = task::Status::success(),
                                            .suspend = true};
  }
}

} // namespace rund::node
