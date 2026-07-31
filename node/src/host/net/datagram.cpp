#include <rund/net/io.hpp>

#include "../../runtime/platform/net.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "address.hpp"
#include "buffer.hpp"
#include "event/record.hpp"
#include "native/result.hpp"
#include "payload/hash.hpp"
#include "ready/ticket.hpp"
#include "registry/socket.hpp"

namespace rund::net::datagram {
namespace {

[[nodiscard]] ReceiveResult fail_receive(
    const ::rund::ReasonCode code, const int err = 0,
    const Address peer = Address{},
    const ::rund::StableHash peer_hash = ::rund::StableHash{}) noexcept {
  ReceiveResult result{code};
  result.native_error = err;
  result.peer = peer;
  result.peer_hash = peer_hash;
  return result;
}

[[nodiscard]] SendResult
fail_send(const ::rund::ReasonCode code, const int err = 0,
          const ::rund::StableHash peer_hash = ::rund::StableHash{}) noexcept {
  SendResult result{code};
  result.native_error = err;
  result.peer_hash = peer_hash;
  return result;
}

[[nodiscard]] std::uint32_t DatagramCapacityBytes() noexcept {
  return node::scheduler_access::ActiveLimits().net_datagram_capacity_bytes;
}

[[nodiscard]] bool ExceedsDatagramCapacity(const std::size_t size) noexcept {
  return size > static_cast<std::size_t>(DatagramCapacityBytes());
}

[[nodiscard]] ReceiveResult
complete_receive(const std::uint64_t socket_id,
                 const std::span<std::byte> buffer,
                 const node::NativeCallResult &native, const Address peer,
                 const ::rund::StableHash peer_hash) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return fail_receive(::rund::ReasonCode::TaskInvalid, native.error, peer,
                        peer_hash);
  }
  (void)RecordNetIngressEvent(
      NetEventRequest{
          .kind = ::rund::host::EventKind::NetRecvDatagram,
          .socket_id = socket_id,
          .native = native,
          .requested_bytes = static_cast<std::uint64_t>(buffer.size()),
          .name_hash = peer_hash,
      },
      std::span<const std::byte>{buffer.data(), buffer.size()});
  if (native.value < 0) {
    return fail_receive(CodeForNative(native), native.error, peer, peer_hash);
  }
  ReceiveResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  result.peer = peer;
  result.peer_hash = peer_hash;
  return result;
}

[[nodiscard]] SendResult
complete_send(const std::uint64_t socket_id,
              const std::span<const std::byte> buffer,
              const node::NativeCallResult &native,
              const ::rund::StableHash peer_hash) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return fail_send(::rund::ReasonCode::TaskInvalid, native.error, peer_hash);
  }
  (void)RecordNetEvent(NetEventRequest{
      .kind = ::rund::host::EventKind::NetSendDatagram,
      .socket_id = socket_id,
      .native = native,
      .requested_bytes = static_cast<std::uint64_t>(buffer.size()),
      .name_hash = peer_hash,
      .payload_hash = PayloadHashForNative(
          native, buffer.data(), static_cast<std::uint64_t>(buffer.size())),
  });
  if (native.value < 0) {
    return fail_send(CodeForNative(native), native.error, peer_hash);
  }
  SendResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  result.peer_hash = peer_hash;
  return result;
}

[[nodiscard]] ::rund::ReasonCode
Shape(const std::span<std::byte> buffer) noexcept {
  if (InvalidBuffer(buffer.data(), buffer.size()) ||
      ExceedsDatagramCapacity(buffer.size())) {
    return ::rund::ReasonCode::TaskInvalid;
  }
  return ::rund::ReasonCode::Ok;
}

[[nodiscard]] ::rund::ReasonCode Shape(const std::span<const std::byte> buffer,
                                       const Address peer) noexcept {
  if (InvalidBuffer(buffer.data(), buffer.size()) ||
      ExceedsDatagramCapacity(buffer.size()) || !peer.valid()) {
    return ::rund::ReasonCode::TaskInvalid;
  }
  return ::rund::ReasonCode::Ok;
}

} // namespace

namespace detail {

ready::Wait prepare(const SocketView socket,
                    const std::span<std::byte> buffer) noexcept {
  const ::rund::ReasonCode shape = Shape(buffer);
  return shape == ::rund::ReasonCode::Ok
             ? ready::read(socket)
             : ready::detail::fail(socket, ready::Interest::Readable, shape);
}

ready::Wait prepare(const SocketView socket,
                    const std::span<const std::byte> buffer,
                    const Address peer) noexcept {
  const ::rund::ReasonCode shape = Shape(buffer, peer);
  return shape == ::rund::ReasonCode::Ok
             ? ready::write(socket)
             : ready::detail::fail(socket, ready::Interest::Writable, shape);
}

} // namespace detail

namespace {

[[nodiscard]] ReceiveResult
receive_prepared(const ready::detail::Claim &claim,
                 const std::span<std::byte> buffer) noexcept {
  node::NativeAddressResult native{};
  std::uint64_t socket_id = 0u;
  {
    ready::detail::Operation operation = ready::detail::prepare(claim);
    if (!operation) {
      return fail_receive(operation.code());
    }
    socket_id = operation.id();
    native = node::NativeRecvFrom(operation.native(), buffer);
  }
  const Address peer = native.call.value >= 0 ? native.address : Address{};
  const ::rund::StableHash peer_hash =
      native.call.value >= 0 ? HashAddress(peer) : ::rund::StableHash{};
  return complete_receive(socket_id, buffer, native.call, peer, peer_hash);
}

[[nodiscard]] SendResult send_prepared(const ready::detail::Claim &claim,
                                       const std::span<const std::byte> buffer,
                                       const Address peer) noexcept {
  const ::rund::StableHash peer_hash = HashAddress(peer);
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    ready::detail::Operation operation = ready::detail::prepare(claim);
    if (!operation) {
      return fail_send(operation.code(), 0, peer_hash);
    }
    socket_id = operation.id();
    native = node::NativeSendTo(operation.native(), buffer, peer);
  }
  if (native.state == node::NativeCallState::InvalidInput) {
    return fail_send(::rund::ReasonCode::TaskInvalid, 0, peer_hash);
  }
  return complete_send(socket_id, buffer, native, peer_hash);
}

} // namespace

ReceiveResult receive(ready::Ticket &&ticket,
                      const std::span<std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return fail_receive(claim.code);
  }
  const ::rund::ReasonCode shape = Shape(buffer);
  if (shape != ::rund::ReasonCode::Ok) {
    return fail_receive(shape);
  }
  return receive_prepared(claim, buffer);
}

SendResult send(ready::Ticket &&ticket, const std::span<const std::byte> buffer,
                const Address peer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return fail_send(claim.code);
  }
  const ::rund::ReasonCode shape = Shape(buffer, peer);
  if (shape != ::rund::ReasonCode::Ok) {
    return fail_send(shape);
  }
  return send_prepared(claim, buffer, peer);
}

ReceiveResult detail::consume(ready::Ticket &&ticket,
                              const std::span<std::byte> buffer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return fail_receive(claim.code);
  }
  return receive_prepared(claim, buffer);
}

SendResult detail::consume(ready::Ticket &&ticket,
                           const std::span<const std::byte> buffer,
                           const Address peer) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return fail_send(claim.code);
  }
  return send_prepared(claim, buffer, peer);
}

} // namespace rund::net::datagram
