#include "../state/storage.hpp"
#include "record.hpp"

namespace rund::node {

::rund::host::Event MakeReactorHostEvent(
    const ReasonCode code,
    const std::uint64_t task_id,
    const std::uint64_t host_handle_id) noexcept {
  return ::rund::host::Event{
      .kind = ::rund::host::EventKind::IoReady,
      .status = code == ReasonCode::Ok ? ::rund::host::Status::Ok
                                       : ::rund::host::Status::SyscallFailed,
      .task_id = task_id,
      .host_handle_id = host_handle_id,
      .native_errno = code == ReasonCode::Ok ? 0 : 1,
  };
}

void Scheduler::RecordReactorObservation(
    const task::ObservationKind kind,
    const ReasonCode code,
    const std::uint64_t task_id,
    const std::uint64_t wait_id,
    const int fd,
    const short interest,
    const short revents) noexcept {
  RecordObservation(kind, code, task_id, wait_id, fd, interest, revents);
}

bool Scheduler::RecordReactorHostEvent(
    const ReasonCode code,
    const std::uint64_t task_id,
    const std::uint64_t host_handle_id) noexcept {
  return RecordHostEvent(
      MakeReactorHostEvent(code, task_id, host_handle_id));
}

}  // namespace rund::node
