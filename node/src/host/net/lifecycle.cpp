#include <rund/net/socket.hpp>

#include <rund/host/event.hpp>

#include "../../runtime/platform/io.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "../../runtime/task/scheduler/reactor/backend.hpp"
#include "../../runtime/task/scheduler/reactor/close.hpp"
#include "../../runtime/task/scheduler/state.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund {
namespace {

[[nodiscard]] ::rund::net::CloseResult
FailSocketClose(const ::rund::ReasonCode code, const int err = 0) noexcept {
  ::rund::net::CloseResult result{code};
  result.native_error = err;
  return result;
}

[[nodiscard]] ::rund::host::Event
MakeSocketCloseEvent(const std::uint64_t socket_id,
                     const node::NativeIoResult native) noexcept {
  return ::rund::host::Event{
      .kind = ::rund::host::EventKind::IoClose,
      .status = native.value >= 0 ? ::rund::host::Status::Ok
                                  : ::rund::host::Status::SyscallFailed,
      .host_handle_id = socket_id,
      .native_errno = native.err,
  };
}

[[nodiscard]] ::rund::net::CloseResult
CompleteNativeSocketClose(const node::NativeIoResult native) noexcept {
  if (native.value < 0) {
    return FailSocketClose(native.unsupported
                               ? ::rund::ReasonCode::IoUnsupported
                               : ::rund::ReasonCode::IoSyscallFailed,
                           native.err);
  }
  return ::rund::net::CloseResult{::rund::ReasonCode::Ok};
}

} // namespace
} // namespace rund

namespace rund::net {

Socket::~Socket() noexcept {
  if (valid()) {
    static_cast<void>(close());
  }
}

Socket &Socket::operator=(Socket &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (valid()) {
    static_cast<void>(close());
  }
  slot_ = std::exchange(other.slot_, nullptr);
  generation_ = std::exchange(other.generation_, 0u);
  return *this;
}

CloseResult Socket::close() noexcept {
  const SocketView socket = view();
  slot_ = nullptr;
  generation_ = 0u;

  if (!BeginSocketClose(socket)) {
    return FailSocketClose(::rund::ReasonCode::IoFdInvalid);
  }
  const int native_socket = detail::SocketAccess::native(socket);
  const std::uint64_t socket_id = detail::SocketAccess::id(native_socket);

  if (InActiveSchedulerTask()) {
    node::Scheduler *const scheduler =
        node::scheduler_access::ActiveScheduler();
    if (scheduler != nullptr) {
      return scheduler->CloseSocket(socket, native_socket);
    }
  }

  const node::NativeIoResult native = node::NativeClose(native_socket);
  (void)RecordHostEvent(MakeSocketCloseEvent(socket_id, native));
  FinishSocketClose(socket);
  return CompleteNativeSocketClose(native);
}

} // namespace rund::net

namespace rund::node {

::rund::net::CloseResult
Scheduler::CloseSocket(const ::rund::net::SocketView socket,
                       const int native_socket) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();

  const auto finish = [this](::rund::net::CloseResult result) noexcept {
    CompletePrimitiveCommit();
    return result;
  };

  const ::rund::ReasonCode invalidation =
      ReactorCloseInvalidateFd(*this, native_socket);
  const node::NativeIoResult native = node::NativeClose(native_socket);
  (void)RecordHostEvent(MakeSocketCloseEvent(
      ::rund::net::detail::SocketAccess::id(native_socket), native));
  FinishSocketClose(socket);
  if (invalidation != ::rund::ReasonCode::Ok) {
    return finish(FailSocketClose(invalidation, native.err));
  }
  return finish(CompleteNativeSocketClose(native));
}

} // namespace rund::node
