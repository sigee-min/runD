#include "scheduler.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "../../runtime/task/scheduler/host.hpp"
#include "../../runtime/task/scheduler/state.hpp"
#include "registry/socket.hpp"
#include "socket/access.hpp"

namespace rund::net {

bool InActiveSchedulerTask() noexcept {
  return node::scheduler_host::ActiveTask();
}

bool RecordHostEvent(::rund::host::Event event) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  return scheduler != nullptr && scheduler->RecordHostEventFromHostApi(event);
}

::rund::detail::task::IoDecision WaitReactor(const SocketView socket,
                                             const short interest) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ::rund::detail::task::IoDecision{
        .status = task::Status::fail(::rund::ReasonCode::NodeRuntimeMissing)};
  }
  return WaitReactor(*scheduler, socket, interest);
}

::rund::detail::task::IoDecision WaitReactor(node::Scheduler &scheduler,
                                             const SocketView socket,
                                             const short interest) noexcept {
  SocketLease lease = LeaseSocket(socket);
  if (!lease) {
    return ::rund::detail::task::IoDecision{
        .status = task::Status::fail(::rund::ReasonCode::IoFdInvalid)};
  }
  return scheduler.WaitReactor(lease.native(), interest, lease.id(),
                               detail::SocketAccess::generation(socket),
                               socket);
}

::rund::detail::task::IoDecision
WaitReactorTimed(const SocketView socket, const short interest,
                 const std::chrono::nanoseconds timeout,
                 const std::uint64_t stop_scheduler_id,
                 const std::uint64_t stop_source_id,
                 const std::uint64_t stop_generation,
                 const std::uint64_t stop_epoch) noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return ::rund::detail::task::IoDecision{
        .status = task::Status::fail(::rund::ReasonCode::NodeRuntimeMissing)};
  }
  return WaitReactorTimed(*scheduler, socket, interest, timeout,
                          stop_scheduler_id, stop_source_id, stop_generation,
                          stop_epoch);
}

::rund::detail::task::IoDecision
WaitReactorTimed(node::Scheduler &scheduler, const SocketView socket,
                 const short interest, const std::chrono::nanoseconds timeout,
                 const std::uint64_t stop_scheduler_id,
                 const std::uint64_t stop_source_id,
                 const std::uint64_t stop_generation,
                 const std::uint64_t stop_epoch) noexcept {
  SocketLease lease = LeaseSocket(socket);
  if (!lease) {
    return ::rund::detail::task::IoDecision{
        .status = task::Status::fail(::rund::ReasonCode::IoFdInvalid)};
  }
  return scheduler.WaitReactorTimed(
      lease.native(), interest, timeout, lease.id(),
      detail::SocketAccess::generation(socket), stop_scheduler_id,
      stop_source_id, stop_generation, stop_epoch, socket);
}

} // namespace rund::net
