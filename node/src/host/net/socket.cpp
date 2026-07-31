#include <rund/net/socket.hpp>

#include <rund/host/event.hpp>

#include "../../runtime/platform/io.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] NonblockingResult FailNonblocking(const ::rund::ReasonCode code,
                                                const int err = 0) noexcept {
  NonblockingResult result{code};
  result.native_error = err;
  return result;
}

} // namespace

std::uint64_t SocketView::id() const noexcept {
  return detail::SocketAccess::id(detail::SocketAccess::native(*this));
}

std::uint64_t Socket::id() const noexcept { return view().id(); }

NonblockingResult nonblocking(const SocketView socket,
                              const bool enabled) noexcept {
  node::NativeIoResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailNonblocking(::rund::ReasonCode::IoFdInvalid);
    }
    socket_id = lease.id();
    native = node::NativeSetNonblockingFd(lease.native(), enabled);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::IoSetNonblocking,
      .status = native.value >= 0 ? ::rund::host::Status::Ok
                                  : ::rund::host::Status::SyscallFailed,
      .host_handle_id = socket_id,
      .completed_bytes = enabled ? 1u : 0u,
      .native_errno = native.err,
  });
  if (native.unsupported) {
    NonblockingResult result =
        FailNonblocking(::rund::ReasonCode::IoUnsupported);
    result.enabled = enabled;
    return result;
  }
  if (native.err != 0) {
    NonblockingResult result =
        FailNonblocking(::rund::ReasonCode::IoSyscallFailed, native.err);
    result.enabled = enabled;
    return result;
  }
  NonblockingResult result{::rund::ReasonCode::Ok};
  result.enabled = enabled;
  return result;
}

} // namespace rund::net
