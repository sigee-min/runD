#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <cerrno>

#include "../../net.hpp"
#include "../address.hpp"
#include "result.hpp"

namespace rund::node {
namespace {

enum class NativeOptionKind {
  Boolean,
  ByteCount,
  Unavailable,
};

struct NativeOptionSpec {
  int level = 0;
  int name = 0;
  NativeOptionKind kind = NativeOptionKind::Unavailable;
};

[[nodiscard]] NativeOptionSpec
SocketOptionSpec(const ::rund::net::option::Name option) noexcept {
  switch (option) {
  case ::rund::net::option::Name::ReuseAddress:
    return NativeOptionSpec{.level = SOL_SOCKET,
                            .name = SO_REUSEADDR,
                            .kind = NativeOptionKind::Boolean};
  case ::rund::net::option::Name::ReusePort:
#ifdef SO_REUSEPORT
    return NativeOptionSpec{.level = SOL_SOCKET,
                            .name = SO_REUSEPORT,
                            .kind = NativeOptionKind::Boolean};
#else
    return NativeOptionSpec{};
#endif
  case ::rund::net::option::Name::TcpNoDelay:
#ifdef TCP_NODELAY
    return NativeOptionSpec{.level = IPPROTO_TCP,
                            .name = TCP_NODELAY,
                            .kind = NativeOptionKind::Boolean};
#else
    return NativeOptionSpec{};
#endif
  case ::rund::net::option::Name::ReceiveBufferBytes:
    return NativeOptionSpec{.level = SOL_SOCKET,
                            .name = SO_RCVBUF,
                            .kind = NativeOptionKind::ByteCount};
  case ::rund::net::option::Name::SendBufferBytes:
    return NativeOptionSpec{.level = SOL_SOCKET,
                            .name = SO_SNDBUF,
                            .kind = NativeOptionKind::ByteCount};
  case ::rund::net::option::Name::IPv6Only:
#ifdef IPV6_V6ONLY
    return NativeOptionSpec{.level = IPPROTO_IPV6,
                            .name = IPV6_V6ONLY,
                            .kind = NativeOptionKind::Boolean};
#else
    return NativeOptionSpec{};
#endif
  }
  return NativeOptionSpec{};
}

[[nodiscard]] NativeCallResult UnavailableSocketOption() noexcept {
  return NativeCallResult{
      .value = -1, .error = ENOPROTOOPT, .state = NativeCallState::Unsupported};
}

[[nodiscard]] int NativeSetValue(const NativeOptionSpec spec,
                                 const ::rund::net::option::Value value) noexcept {
  switch (spec.kind) {
  case NativeOptionKind::Boolean:
    return value.flag ? 1 : 0;
  case NativeOptionKind::ByteCount:
    return value.bytes;
  case NativeOptionKind::Unavailable:
    return 0;
  }
  return 0;
}

} // namespace

NativeCallResult
NativeSetSocketOption(const int fd, const ::rund::net::option::Name option,
                      const ::rund::net::option::Value value) noexcept {
  const NativeOptionSpec spec = SocketOptionSpec(option);
  if (spec.kind == NativeOptionKind::Unavailable) {
    return UnavailableSocketOption();
  }
  const int native_value = NativeSetValue(spec, value);
  errno = 0;
  const int result = ::setsockopt(fd, spec.level, spec.name, &native_value,
                                  sizeof(native_value));
  return PosixNetResult(result < 0 ? -1 : 0, result < 0 ? errno : 0);
}

NativeCallResult
NativeGetSocketOption(const int fd, const ::rund::net::option::Name option) noexcept {
  const NativeOptionSpec spec = SocketOptionSpec(option);
  if (spec.kind == NativeOptionKind::Unavailable) {
    return UnavailableSocketOption();
  }
  int native_value = 0;
  socklen_t length = sizeof(native_value);
  errno = 0;
  const int result =
      ::getsockopt(fd, spec.level, spec.name, &native_value, &length);
  return PosixNetResult(result < 0 ? -1 : native_value, result < 0 ? errno : 0);
}

NativeCallResult NativeGetSocketError(const int fd,
                                      const ::rund::net::Address &address) noexcept {
  sockaddr_storage storage{};
  socklen_t address_length = 0;
  if (!PosixAddress(address, &storage, &address_length)) {
    return PosixNetResult(-1, EINVAL, true);
  }
  int socket_error = 0;
  socklen_t length = sizeof(socket_error);
  errno = 0;
  const int result =
      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &length);
  if (result < 0) {
    return PosixNetResult(-1, errno);
  }
  return PosixNetResult(socket_error == 0 ? 0 : -1, socket_error);
}

} // namespace rund::node
