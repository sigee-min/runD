#include <rund/net/options.hpp>

#include <rund/host/event.hpp>

#include "../../runtime/platform/net.hpp"
#include "native/result.hpp"
#include "registry/socket.hpp"
#include "scheduler.hpp"

#include <cstdint>
#include <limits>

namespace rund::net::option {
namespace {

[[nodiscard]] Result fail(const Name option, const ::rund::ReasonCode code,
                          const int err = 0,
                          const Value value = Value{}) noexcept {
  Result result{code};
  result.option = option;
  result.value = value;
  result.native_error = err;
  return result;
}

[[nodiscard]] bool known(const Name option) noexcept {
  switch (option) {
  case Name::ReuseAddress:
  case Name::ReusePort:
  case Name::TcpNoDelay:
  case Name::ReceiveBufferBytes:
  case Name::SendBufferBytes:
  case Name::IPv6Only:
    return true;
  }
  return false;
}

[[nodiscard]] bool is_byte(const Name option) noexcept {
  return option == Name::ReceiveBufferBytes || option == Name::SendBufferBytes;
}

[[nodiscard]] bool valid_value(const Name option, const Value value) noexcept {
  if (is_byte(option)) {
    return value.bytes > 0;
  }
  return true;
}

[[nodiscard]] std::uint64_t id(const Name option) noexcept {
  return static_cast<std::uint64_t>(option);
}

[[nodiscard]] std::uint64_t normalized(const Name option,
                                       const Value value) noexcept {
  if (is_byte(option)) {
    return value.bytes > 0 ? static_cast<std::uint64_t>(value.bytes) : 0u;
  }
  return value.flag ? 1u : 0u;
}

[[nodiscard]] Value from_native(const Name option,
                                const node::NativeCallResult &native) noexcept {
  Value value{};
  if (native.value < 0) {
    return value;
  }
  if (is_byte(option)) {
    const std::int64_t bounded =
        native.value > static_cast<std::int64_t>(
                           std::numeric_limits<std::int32_t>::max())
            ? static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max())
            : native.value;
    value.bytes = static_cast<std::int32_t>(bounded);
  } else {
    value.flag = native.value != 0;
  }
  return value;
}

} // namespace

Result set(const SocketView socket, const Name option,
           const Value value) noexcept {
  if (!known(option)) {
    return fail(option, ::rund::ReasonCode::TaskInvalid, 0, value);
  }
  if (!valid_value(option, value)) {
    return fail(option, ::rund::ReasonCode::TaskInvalid, 0, value);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return fail(option, ::rund::ReasonCode::IoFdInvalid, 0, value);
    }
    socket_id = lease.id();
    native = node::NativeSetSocketOption(lease.native(), option, value);
  }
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetSetSocketOption,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .offset = id(option),
      .requested_bytes = normalized(option, value),
      .completed_bytes = normalized(option, value),
      .native_errno = native.error,
  });
  if (native.value < 0) {
    return fail(option, CodeForNative(native), native.error, value);
  }
  Result result{::rund::ReasonCode::Ok};
  result.option = option;
  result.value = value;
  return result;
}

Result get(const SocketView socket, const Name option) noexcept {
  if (!known(option)) {
    return fail(option, ::rund::ReasonCode::TaskInvalid);
  }
  node::NativeCallResult native{};
  std::uint64_t socket_id = 0u;
  {
    SocketLease lease = LeaseSocket(socket);
    if (!lease) {
      return fail(option, ::rund::ReasonCode::IoFdInvalid);
    }
    socket_id = lease.id();
    native = node::NativeGetSocketOption(lease.native(), option);
  }
  const Value value = from_native(option, native);
  (void)RecordHostEvent(::rund::host::Event{
      .kind = ::rund::host::EventKind::NetGetSocketOption,
      .status = StatusForNative(native),
      .host_handle_id = socket_id,
      .offset = id(option),
      .requested_bytes = normalized(option, value),
      .completed_bytes = normalized(option, value),
      .native_errno = native.error,
  });
  if (native.value < 0) {
    return fail(option, CodeForNative(native), native.error, value);
  }
  Result result{::rund::ReasonCode::Ok};
  result.option = option;
  result.value = value;
  return result;
}

} // namespace rund::net::option
