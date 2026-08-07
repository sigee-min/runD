#include <rund/net/io.hpp>

#include "../../runtime/platform/net.hpp"
#include "buffer.hpp"
#include "bytes.hpp"
#include "event/record.hpp"
#include "native/result.hpp"
#include "payload/hash.hpp"
#include "ready/ticket.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

namespace rund::net {
namespace {

[[nodiscard]] ReceiveResult FailRecv(const ::rund::ReasonCode code,
                                     const int err = 0) noexcept {
  ReceiveResult result{code};
  result.native_error = err;
  return result;
}

[[nodiscard]] SendResult FailSend(const ::rund::ReasonCode code,
                                  const int err = 0) noexcept {
  SendResult result{code};
  result.native_error = err;
  return result;
}

[[nodiscard]] ::rund::ReasonCode
Shape(const std::span<std::byte> buffer) noexcept {
  return InvalidBuffer(buffer.data(), buffer.size())
             ? ::rund::ReasonCode::TaskInvalid
             : ::rund::ReasonCode::Ok;
}

[[nodiscard]] ::rund::ReasonCode
Shape(const std::span<const std::byte> buffer) noexcept {
  return InvalidBuffer(buffer.data(), buffer.size())
             ? ::rund::ReasonCode::TaskInvalid
             : ::rund::ReasonCode::Ok;
}

} // namespace

namespace detail {

ready::Wait prepare(const SocketView socket,
                    const std::span<std::byte> buffer) noexcept {
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return ready::detail::fail(socket, ready::Interest::Readable, shape);
  }
  return buffer.empty()
             ? ready::detail::complete(socket, ready::Interest::Readable)
             : ready::read(socket);
}

ready::Wait prepare(const SocketView socket,
                    const std::span<const std::byte> buffer) noexcept {
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return ready::detail::fail(socket, ready::Interest::Writable, shape);
  }
  return buffer.empty()
             ? ready::detail::complete(socket, ready::Interest::Writable)
             : ready::write(socket);
}

ReceiveResult complete_receive(const std::uint64_t socket_id,
                               const std::span<std::byte> buffer,
                               const node::NativeCallResult &native) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return FailRecv(::rund::ReasonCode::TaskInvalid, native.error);
  }
  (void)RecordNetIngressEvent(
      NetEventRequest{
          .kind = ::rund::host::EventKind::NetRecv,
          .socket_id = socket_id,
          .native = native,
          .requested_bytes = static_cast<std::uint64_t>(buffer.size()),
      },
      std::span<const std::byte>{buffer.data(), buffer.size()});
  if (native.value < 0) {
    ReceiveResult result = FailRecv(CodeForNative(native), native.error);
    return result;
  }
  ReceiveResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  return result;
}

SendResult complete_send(const std::uint64_t socket_id,
                         const std::span<const std::byte> buffer,
                         const node::NativeCallResult &native) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return FailSend(::rund::ReasonCode::TaskInvalid, native.error);
  }
  (void)RecordNetEvent(NetEventRequest{
      .kind = ::rund::host::EventKind::NetSend,
      .socket_id = socket_id,
      .native = native,
      .requested_bytes = static_cast<std::uint64_t>(buffer.size()),
      .payload_hash = PayloadHashForNative(
          native, buffer.data(), static_cast<std::uint64_t>(buffer.size())),
  });
  if (native.value < 0) {
    SendResult result = FailSend(CodeForNative(native), native.error);
    return result;
  }
  SendResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  return result;
}

} // namespace detail

ReceiveResult
detail::receive_attempt(const ready::detail::Claim &claim,
                        const std::span<std::byte> buffer) noexcept {
  ready::detail::Operation operation = ready::detail::prepare(claim);
  if (!operation) {
    return FailRecv(operation.code());
  }
  return detail::complete_receive(
      operation.id(), buffer, node::NativeTryRecv(operation.native(), buffer));
}

SendResult
detail::send_attempt(const ready::detail::Claim &claim,
                     const std::span<const std::byte> buffer) noexcept {
  ready::detail::Operation operation = ready::detail::prepare(claim);
  if (!operation) {
    return FailSend(operation.code());
  }
  return detail::complete_send(operation.id(), buffer,
                               node::NativeTrySend(operation.native(), buffer));
}

namespace direct {

ReceiveResult receive(const SocketView socket,
                      const std::span<std::byte> buffer) noexcept {
  if (InActiveSchedulerTask()) {
    return FailRecv(::rund::ReasonCode::TaskInvalid);
  }
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return FailRecv(shape);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailRecv(::rund::ReasonCode::IoFdInvalid);
    }
    socket_id = lease.id();
    native = node::NativeRecv(lease.native(), buffer);
  }
  return detail::complete_receive(socket_id, buffer, native);
}

SendResult send(const SocketView socket,
                const std::span<const std::byte> buffer) noexcept {
  if (InActiveSchedulerTask()) {
    return FailSend(::rund::ReasonCode::TaskInvalid);
  }
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return FailSend(shape);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return FailSend(::rund::ReasonCode::IoFdInvalid);
    }
    socket_id = lease.id();
    native = node::NativeSend(lease.native(), buffer);
  }
  return detail::complete_send(socket_id, buffer, native);
}

} // namespace direct

ReceiveResult receive(ready::Ticket &&ticket,
                      const std::span<std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return FailRecv(claim.code);
  }
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return FailRecv(shape);
  }
  return detail::receive_attempt(claim, buffer);
}

SendResult send(ready::Ticket &&ticket,
                const std::span<const std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return FailSend(claim.code);
  }
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return FailSend(shape);
  }
  return detail::send_attempt(claim, buffer);
}

ReceiveResult detail::consume(ready::Ticket &&ticket,
                              const std::span<std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return FailRecv(claim.code);
  }
  return detail::receive_attempt(claim, buffer);
}

SendResult detail::consume(ready::Ticket &&ticket,
                           const std::span<const std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return FailSend(claim.code);
  }
  return detail::send_attempt(claim, buffer);
}

} // namespace rund::net
