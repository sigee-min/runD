#include "../state/storage.hpp"

namespace rund::node {

::rund::detail::task::IoDecision Scheduler::WaitReactorTimed(
    const int fd, const short interest, const std::chrono::nanoseconds timeout,
    const std::uint64_t host_handle_id, const std::uint64_t fd_generation,
    const ::rund::detail::task::StopIdentity stop,
    const ::rund::net::SocketView socket) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::IoPark);
  EnsureCurrentCommit();

  TaskRecord *record = nullptr;
  ::rund::detail::task::IoDecision result{};
  std::uint64_t wait_host_handle_id = 0u;
  const std::uint64_t task_id = CurrentTaskId();
  if (!ValidateTimedReactorWait(fd, timeout, task_id, host_handle_id,
                                fd_generation, stop, record,
                                wait_host_handle_id, result)) {
    return result;
  }
  if (!ResolveImmediateTimedReactorWait(*record, fd, interest, timeout,
                                        wait_host_handle_id, result)) {
    return result;
  }

  std::uint64_t wait_id = 0u;
  if (!ParkTimedReactorWait(*record, fd, interest, timeout, wait_host_handle_id,
                            fd_generation, stop.source(), socket, wait_id,
                            result)) {
    return result;
  }
  return ResumeTimedReactorWait(*record, wait_id);
}

} // namespace rund::node
