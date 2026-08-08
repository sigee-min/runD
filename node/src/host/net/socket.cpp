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
  SocketLease lease = LeaseSocket(socket);
  if (!lease) {
    return FailNonblocking(::rund::ReasonCode::IoFdInvalid);
  }
  const std::uint64_t socket_id = lease.id();
  const node::NativeIoResult native =
      node::NativeSetNonblockingFd(lease.native(), enabled);
  lease = SocketLease{};

  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::IoSetNonblocking,
      .status = native.disposition() == node::NativeIoDisposition::Complete
                    ? ::rund::host::Status::Ok
                    : ::rund::host::Status::SyscallFailed,
      .host_handle_id = socket_id,
      .completed_bytes = enabled ? 1u : 0u,
      .native_errno = native.native_error(),
  });

  switch (native.disposition()) {
  case node::NativeIoDisposition::Complete: {
    NonblockingResult result{::rund::ReasonCode::Ok};
    result.enabled = enabled;
    return result;
  }
  case node::NativeIoDisposition::Unsupported: {
    NonblockingResult result =
        FailNonblocking(::rund::ReasonCode::IoUnsupported);
    result.enabled = enabled;
    return result;
  }
  case node::NativeIoDisposition::InvalidBuffer:
  case node::NativeIoDisposition::Failed: {
    NonblockingResult result = FailNonblocking(
        ::rund::ReasonCode::IoSyscallFailed, native.native_error());
    result.enabled = enabled;
    return result;
  }
  }
  std::abort();
}

} // namespace rund::net
