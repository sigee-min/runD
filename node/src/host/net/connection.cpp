#include <rund/net/accept.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/server/accept.hpp>

#include <rund/host/event.hpp>

#include "../../runtime/platform/io.hpp"
#include "../../runtime/platform/net.hpp"
#include "address.hpp"
#include "native/result.hpp"
#include "ready/ticket.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] accept::Result fail_accept(const ::rund::ReasonCode code,
                                         const int err = 0) noexcept {
  accept::Result result{code};
  result.native_error = err;
  return result;
}

[[nodiscard]] accept::Result
CompleteAccept(const std::uint64_t listener_id,
               const node::NativeAddressResult native) noexcept {
  const ::rund::StableHash peer_hash = native.call.value >= 0
                                           ? HashAddress(native.address)
                                           : ::rund::StableHash{};
  Socket accepted{};
  if (native.call.value >= 0) {
    SocketAdmission admission =
        AdmitNativeSocket(static_cast<int>(native.call.value));
    if (!admission) {
      const node::NativeIoResult closed =
          node::NativeClose(static_cast<int>(native.call.value));
      (void)closed;
      return fail_accept(admission.code());
    }
    accepted = std::move(admission).take_socket();
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetAccept,
      .status = StatusForNative(native.call),
      .host_handle_id = listener_id,
      .offset = accepted.id(),
      .native_errno = native.call.error,
      .payload_hash = peer_hash,
  });
  if (native.call.value < 0) {
    return fail_accept(CodeForNative(native.call), native.call.error);
  }
  accept::Result result{::rund::ReasonCode::Ok};
  result.socket = std::move(accepted);
  result.peer_hash = peer_hash;
  return result;
}

template <typename Lease>
[[nodiscard]] accept::Result
AttemptAccept(Lease lease, const ::rund::ReasonCode invalid) noexcept {
  if (!lease) {
    return fail_accept(invalid);
  }
  if (!node::NativeIsNonblockingFd(lease.native())) {
    return fail_accept(::rund::ReasonCode::TaskInvalid);
  }
  return CompleteAccept(lease.id(), node::NativeAccept(lease.native()));
}

[[nodiscard]] accept::Result
AttemptAccept(const ready::detail::Claim &claim) noexcept {
  ready::detail::Operation operation = ready::detail::prepare(claim);
  const ::rund::ReasonCode invalid = operation.code();
  return AttemptAccept(std::move(operation), invalid);
}

[[nodiscard]] accept::Result AttemptAccept(const SocketView listener) noexcept {
  return AttemptAccept(LeaseSocket(listener), ::rund::ReasonCode::IoFdInvalid);
}

[[nodiscard]] connect::Result fail_connect(
    const ::rund::ReasonCode code, const int err = 0,
    const ::rund::StableHash address_hash = ::rund::StableHash{}) noexcept {
  connect::Result result{code};
  result.native_error = err;
  result.address_hash = address_hash;
  return result;
}

} // namespace

accept::Result accept::one(ready::Ticket &&ticket) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return fail_accept(claim.code);
  }
  return AttemptAccept(claim);
}

accept::Result server::detail::next(const SocketView listener) noexcept {
  return AttemptAccept(listener);
}

accept::Drain
accept::detail::drain(ready::Ticket &&ticket, const accept::Budget budget,
                      const void *const state,
                      const accept::detail::Handler handler) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return accept::fail_drain(claim.code);
  }
  if (handler == nullptr) {
    return accept::fail_drain(::rund::ReasonCode::TaskInvalid);
  }
  if (budget.max_accepts == 0u) {
    accept::Drain result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }
  accept::Drain result{::rund::ReasonCode::Ok};
  for (std::uint32_t attempt = 0u; attempt < budget.max_accepts; ++attempt) {
    accept::Result accepted = AttemptAccept(claim);
    if (!accepted) {
      if (accepted.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, accepted.code());
      return result;
    }
    ++result.accepts;
    if (!handler(state, std::move(accepted))) {
      result.handler_stopped = true;
      return result;
    }
  }
  result.budget_exhausted = true;
  return result;
}

connect::Result connect::start(const SocketView socket,
                               const Address address) noexcept {
  const ::rund::StableHash address_hash = HashAddress(address);
  if (!address.valid()) {
    return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return fail_connect(::rund::ReasonCode::IoFdInvalid, 0, address_hash);
    }
    if (!node::NativeIsNonblockingFd(lease.native())) {
      return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
    }
    socket_id = lease.id();
    native = node::NativeConnect(lease.native(), address);
  }
  if (native.state == node::NativeCallState::InvalidInput) {
    return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetConnect,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .native_errno = native.error,
      .payload_hash = address_hash,
  });
  if (native.state != node::NativeCallState::Complete &&
      native.state != node::NativeCallState::InProgress) {
    connect::Result result =
        fail_connect(CodeForNative(native), native.error, address_hash);
    return result;
  }
  connect::Result result{::rund::ReasonCode::Ok};
  result.native_error = native.error;
  result.address_hash = address_hash;
  return result;
}

connect::Result connect::finish(ready::Ticket &&ticket,
                                const Address address) noexcept {
  const ::rund::StableHash address_hash = HashAddress(address);
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return fail_connect(claim.code, 0, address_hash);
  }
  if (!address.valid()) {
    return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    ready::detail::Operation operation = ready::detail::prepare(claim);
    if (!operation) {
      return fail_connect(operation.code(), 0, address_hash);
    }
    if (!node::NativeIsNonblockingFd(operation.native())) {
      return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
    }
    socket_id = operation.id();
    native = node::NativeGetSocketError(operation.native(), address);
  }
  if (native.state == node::NativeCallState::InvalidInput) {
    return fail_connect(::rund::ReasonCode::TaskInvalid, 0, address_hash);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetConnect,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .native_errno = native.error,
      .payload_hash = address_hash,
  });
  if (native.value < 0) {
    connect::Result result =
        fail_connect(CodeForNative(native), native.error, address_hash);
    return result;
  }
  connect::Result result{::rund::ReasonCode::Ok};
  result.address_hash = address_hash;
  return result;
}

} // namespace rund::net
