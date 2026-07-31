#include <rund/net/vectored.hpp>

#include "../../runtime/platform/net.hpp"
#include "../../runtime/platform/net/vectored.hpp"
#include "../../runtime/task/scheduler/access.hpp"
#include "event/record.hpp"
#include "native/result.hpp"
#include "payload/hash.hpp"
#include "ready/ticket.hpp"
#include "registry/socket.hpp"
#include "vectored.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <rund/session/scheduler.hpp>

namespace rund::net::batch {
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

[[nodiscard]] node::NativeVectoredResult EmptyBatch() noexcept {
  return node::NativeVectoredResult{
      .call =
          {
              .value = 0,
              .error = 0,
              .state = node::NativeCallState::Complete,
          },
      .admitted_bytes = 0u,
  };
}

} // namespace

namespace detail {

ReceiveResult complete_receive(const std::uint64_t socket_id,
                               const std::span<const Buffer> slices,
                               const std::uint64_t admitted_bytes,
                               const node::NativeCallResult &native) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return FailRecv(::rund::ReasonCode::TaskInvalid, native.error);
  }
  (void)RecordNetIngressEvent(
      NetEventRequest{
          .kind = ::rund::host::EventKind::NetRecvVectored,
          .socket_id = socket_id,
          .native = native,
          .requested_bytes = admitted_bytes,
      },
      slices);
  if (native.value < 0) {
    return FailRecv(CodeForNative(native), native.error);
  }
  ReceiveResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  return result;
}

SendResult complete_send(const std::uint64_t socket_id,
                         const std::span<const Slice> slices,
                         const std::uint64_t admitted_bytes,
                         const node::NativeCallResult &native) noexcept {
  if (native.state == node::NativeCallState::InvalidInput) {
    return FailSend(::rund::ReasonCode::TaskInvalid, native.error);
  }
  const std::uint64_t completed = CompletedByteCount(native, admitted_bytes);
  (void)RecordNetEvent(NetEventRequest{
      .kind = ::rund::host::EventKind::NetSendVectored,
      .socket_id = socket_id,
      .native = native,
      .requested_bytes = admitted_bytes,
      .payload_hash = native.value < 0
                          ? ::rund::StableHash{}
                          : PayloadHashForPrefix(slices, completed),
  });
  if (native.value < 0) {
    return FailSend(CodeForNative(native), native.error);
  }
  SendResult result{::rund::ReasonCode::Ok};
  result.bytes = native.value;
  return result;
}

} // namespace detail

ReceiveResult receive(ready::Ticket &&ticket,
                      const std::span<const Buffer> slices) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return FailRecv(claim.code);
  }
  const node::nativeio::VectoredBatch batch = node::nativeio::PrepareSlices(
      slices,
      std::min<std::size_t>(
          node::scheduler_access::ActiveLimits().net_iov_capacity,
          ::rund::SchedulerConfig{}.net_iov_capacity),
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()));
  if (!batch.valid) {
    return FailRecv(::rund::ReasonCode::TaskInvalid);
  }
  node::NativeVectoredResult native{};
  std::uint64_t socket_id = 0u;
  {
    ready::detail::Operation operation = ready::detail::prepare(claim);
    if (!operation) {
      return FailRecv(operation.code());
    }
    socket_id = operation.id();
    native = batch.admitted_bytes == 0u
                 ? EmptyBatch()
                 : node::NativeRecvVectored(operation.native(), batch);
  }
  return detail::complete_receive(socket_id, slices, native.admitted_bytes,
                                  native.call);
}

SendResult send(ready::Ticket &&ticket,
                const std::span<const Slice> slices) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return FailSend(claim.code);
  }
  const node::nativeio::VectoredBatch batch = node::nativeio::PrepareSlices(
      slices,
      std::min<std::size_t>(
          node::scheduler_access::ActiveLimits().net_iov_capacity,
          ::rund::SchedulerConfig{}.net_iov_capacity),
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()));
  if (!batch.valid) {
    return FailSend(::rund::ReasonCode::TaskInvalid);
  }
  node::NativeVectoredResult native{};
  std::uint64_t socket_id = 0u;
  {
    ready::detail::Operation operation = ready::detail::prepare(claim);
    if (!operation) {
      return FailSend(operation.code());
    }
    socket_id = operation.id();
    native = batch.admitted_bytes == 0u
                 ? EmptyBatch()
                 : node::NativeSendVectored(operation.native(), batch);
  }
  return detail::complete_send(socket_id, slices, native.admitted_bytes,
                               native.call);
}

} // namespace rund::net::batch
