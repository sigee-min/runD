#include <rund/net/listener.hpp>

#include <rund/host/event.hpp>
#include <rund/net/socket.hpp>

#include "../../runtime/platform/io.hpp"
#include "../../runtime/platform/net.hpp"
#include "address.hpp"
#include "native/result.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] OpenResult FailOpenSocket(
    const ::rund::ReasonCode code, const int err = 0, Socket socket = Socket{},
    const NonblockingResult nonblocking = NonblockingResult{}) noexcept {
  OpenResult result{code};
  result.socket = std::move(socket);
  result.native_error = err;
  result.nonblocking = nonblocking;
  return result;
}

[[nodiscard]] BindResult FailBind(
    const ::rund::ReasonCode code, const int err = 0,
    const ::rund::StableHash address_hash = ::rund::StableHash{}) noexcept {
  BindResult result{code};
  result.native_error = err;
  result.address_hash = address_hash;
  return result;
}

[[nodiscard]] ListenResult FailListen(const ::rund::ReasonCode code,
                                      const int err = 0,
                                      const int backlog = 0) noexcept {
  ListenResult result{code};
  result.native_error = err;
  result.backlog = backlog;
  return result;
}

[[nodiscard]] LocalResult FailLocalAddress(
    const ::rund::ReasonCode code, const int err = 0,
    const Address address = Address{},
    const ::rund::StableHash address_hash = ::rund::StableHash{}) noexcept {
  LocalResult result{code};
  result.address = address;
  result.native_error = err;
  result.address_hash = address_hash;
  return result;
}

[[nodiscard]] ShutdownResult
FailShutdown(const ::rund::ReasonCode code, const int err = 0,
             const ShutdownMode mode = ShutdownMode::ReadWrite) noexcept {
  ShutdownResult result{code};
  result.native_error = err;
  result.mode = mode;
  return result;
}

[[nodiscard]] NonblockingResult NonblockingNotRequested() noexcept {
  NonblockingResult result{::rund::ReasonCode::Ok};
  result.enabled = false;
  return result;
}

} // namespace

OpenResult open(const OpenOptions options) noexcept {
  const node::NativeCallResult native =
      node::NativeSocket(options.family, options.transport);
  if (native.value < 0) {
    (void)RecordHostEvent(::rund::host::Event{
        .kind = ::rund::host::EventKind::NetSocket,
        .status = StatusForNative(native),
        .native_errno = native.error,
    });
    return FailOpenSocket(CodeForNative(native), native.error);
  }
  SocketAdmission admission = AdmitNativeSocket(static_cast<int>(native.value));
  if (!admission) {
    const node::NativeIoResult closed =
        node::NativeClose(static_cast<int>(native.value));
    (void)closed;
    return FailOpenSocket(admission.code(), native.error, Socket{},
                          NonblockingResult{});
  }
  Socket socket = std::move(admission).take_socket();
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetSocket,
      .status = StatusForNative(native),
      .host_handle_id = socket.id(),
      .native_errno = native.error,
  });
  NonblockingResult configured = NonblockingNotRequested();
  if (options.nonblocking) {
    configured = nonblocking(socket.view(), true);
    if (!configured) {
      return FailOpenSocket(configured.code(), configured.native_error,
                            Socket{}, configured);
    }
  }

  OpenResult result{::rund::ReasonCode::Ok};
  result.socket = std::move(socket);
  result.nonblocking = configured;
  return result;
}

BindResult bind(const SocketView socket, const Address address) noexcept {
  const ::rund::StableHash address_hash = HashAddress(address);
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailBind(::rund::ReasonCode::IoFdInvalid, 0, address_hash);
    }
    socket_id = lease.id();
    native = node::NativeBind(lease.native(), address);
  }
  if (native.state == node::NativeCallState::InvalidInput) {
    return FailBind(::rund::ReasonCode::TaskInvalid, 0, address_hash);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetBind,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .native_errno = native.error,
      .payload_hash = address_hash,
  });
  if (native.value < 0) {
    return FailBind(CodeForNative(native), native.error, address_hash);
  }
  BindResult result{::rund::ReasonCode::Ok};
  result.address_hash = address_hash;
  return result;
}

ListenResult listen(const SocketView socket, const int backlog) noexcept {
  if (backlog < 0) {
    return FailListen(::rund::ReasonCode::TaskInvalid, 0, backlog);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailListen(::rund::ReasonCode::IoFdInvalid, 0, backlog);
    }
    socket_id = lease.id();
    native = node::NativeListen(lease.native(), backlog);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetListen,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .completed_bytes = static_cast<std::uint64_t>(backlog),
      .native_errno = native.error,
  });
  if (native.value < 0) {
    return FailListen(CodeForNative(native), native.error, backlog);
  }
  ListenResult result{::rund::ReasonCode::Ok};
  result.backlog = backlog;
  return result;
}

LocalResult local(const SocketView socket) noexcept {
  node::NativeAddressResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailLocalAddress(::rund::ReasonCode::IoFdInvalid);
    }
    socket_id = lease.id();
    native = node::NativeGetSockName(lease.native());
  }
  const Address address = native.call.value >= 0 ? native.address : Address{};
  const ::rund::StableHash address_hash =
      native.call.value >= 0 ? HashAddress(address) : ::rund::StableHash{};
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetLocalAddress,
      .status = StatusForNative(native.call),
      .host_handle_id = socket_id,
      .native_errno = native.call.error,
      .payload_hash = address_hash,
  });
  if (native.call.value < 0) {
    return FailLocalAddress(CodeForNative(native.call), native.call.error,
                            address, address_hash);
  }
  if (!address) {
    return FailLocalAddress(::rund::ReasonCode::TaskInvalid, 0, address,
                            address_hash);
  }
  LocalResult result{::rund::ReasonCode::Ok};
  result.address = address;
  result.address_hash = address_hash;
  return result;
}

ShutdownResult shutdown(const SocketView socket,
                        const ShutdownMode mode) noexcept {
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailShutdown(::rund::ReasonCode::IoFdInvalid, 0, mode);
    }
    socket_id = lease.id();
    native = node::NativeShutdown(lease.native(), mode);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetShutdown,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .completed_bytes = static_cast<std::uint64_t>(mode),
      .native_errno = native.error,
  });
  if (native.value < 0) {
    return FailShutdown(CodeForNative(native), native.error, mode);
  }
  ShutdownResult result{::rund::ReasonCode::Ok};
  result.mode = mode;
  return result;
}

} // namespace rund::net
