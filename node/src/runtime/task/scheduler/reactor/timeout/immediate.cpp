#include "../../../../reactor/readiness/handle.hpp"
#include "../../../../reactor/readiness/mask.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"
#include "../poll.hpp"

namespace rund::node {

bool Scheduler::ResolveImmediateTimedReactorWait(
    TaskRecord &record, const int fd, const short interest,
    const std::chrono::nanoseconds timeout,
    const std::uint64_t wait_host_handle_id,
    ::rund::detail::task::IoDecision &result) noexcept {
  const ReactorRequest immediate_request{
      .wait_id = 0u,
      .task_id = record.id,
      .fd = ReactorHandleFromPublic(fd),
      .interest = ReactorInterestFromBits(interest),
  };
  const ReactorProbeResult immediate =
      ReactorProbeNow(state_->reactor.reactor.platform,
                      state_->reactor.reactor.probe_ready, immediate_request);
  if (immediate.failed) {
    if (immediate.unavailable) {
      result = FailIo(ReasonCode::ReactorBackendUnavailable);
      CompletePrimitiveCommit();
      return false;
    }
    RecordReactorObservation(task::ObservationKind::IoPollFailed,
                             ReasonCode::IoPollFailed, record.id, 0u, fd,
                             interest, 0);
    if (!RecordReactorHostEvent(ReasonCode::IoPollFailed, record.id,
                                wait_host_handle_id)) {
      result = FailIo(ReasonCode::HostReplayEventMismatch);
      CompletePrimitiveCommit();
      return false;
    }
    result = FailIo(ReasonCode::IoPollFailed);
    CompletePrimitiveCommit();
    return false;
  }

  const ReactorReady *immediate_ready_event =
      immediate.has_ready ? &immediate.ready : nullptr;
  const ReactorEvent immediate_events = immediate_ready_event == nullptr
                                            ? ReactorEvent::None
                                            : immediate_ready_event->events;
  const short immediate_revents = ReactorEventBits(immediate_events);
  if (immediate_ready_event != nullptr && immediate_ready_event->invalid) {
    RecordReactorObservation(task::ObservationKind::IoInvalid,
                             ReasonCode::IoFdInvalid, record.id, 0u, fd,
                             interest, immediate_revents);
    if (!RecordReactorHostEvent(ReasonCode::IoFdInvalid, record.id,
                                wait_host_handle_id)) {
      result = FailIo(ReasonCode::HostReplayEventMismatch);
      CompletePrimitiveCommit();
      return false;
    }
    result = FailIo(ReasonCode::IoFdInvalid, immediate_revents);
    CompletePrimitiveCommit();
    return false;
  }

  if (ReactorEventsMatch(immediate_events, ReactorInterestFromBits(interest))) {
    RecordReactorObservation(task::ObservationKind::IoReady, ReasonCode::Ok,
                             record.id, 0u, fd, interest, immediate_revents);
    if (!RecordReactorHostEvent(ReasonCode::Ok, record.id,
                                wait_host_handle_id)) {
      result = FailIo(ReasonCode::HostReplayEventMismatch);
      CompletePrimitiveCommit();
      return false;
    }
    result = ::rund::detail::task::IoDecision{.status = task::Status::success(),
                                              .revents = immediate_revents};
    CompletePrimitiveCommit();
    return false;
  }

  if (timeout.count() == 0) {
    result = FailIo(ReasonCode::IoTimedOut);
    CompletePrimitiveCommit();
    return false;
  }
  return true;
}

} // namespace rund::node
