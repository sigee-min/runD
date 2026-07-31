#include "../net.hpp"
#include "address.hpp"
#include "buffer.hpp"
#include "net/result.hpp"
#include "net/socket/call.hpp"

#include <sys/socket.h>

#include <cerrno>

namespace rund::node {

NativeCallResult NativeSocket(const ::rund::net::Family family,
                              const ::rund::net::Transport transport) noexcept {
  const int native_family = family == ::rund::net::Family::IPv4   ? AF_INET
                            : family == ::rund::net::Family::IPv6 ? AF_INET6
                                                                  : -1;
  const int native_type =
      transport == ::rund::net::Transport::Stream     ? SOCK_STREAM
      : transport == ::rund::net::Transport::Datagram ? SOCK_DGRAM
                                                      : -1;
  if (native_family < 0 || native_type < 0) {
    return PosixNetResult(-1, EINVAL, true);
  }
  errno = 0;
  const int value = ::socket(native_family, native_type, 0);
  return PosixNetResult(value, value < 0 ? errno : 0);
}

NativeCallResult NativeBind(const int fd,
                            const ::rund::net::Address &address) noexcept {
  sockaddr_storage storage{};
  socklen_t length = 0;
  if (!PosixAddress(address, &storage, &length)) {
    return PosixNetResult(-1, EINVAL, true);
  }
  errno = 0;
  const int value =
      ::bind(fd, reinterpret_cast<const sockaddr *>(&storage), length);
  return PosixNetResult(value < 0 ? -1 : 0, value < 0 ? errno : 0);
}

NativeCallResult NativeListen(const int fd, const int backlog) noexcept {
  errno = 0;
  const int value = ::listen(fd, backlog);
  return PosixNetResult(value < 0 ? -1 : 0, value < 0 ? errno : 0);
}

NativeAddressResult NativeGetSockName(const int fd) noexcept {
  sockaddr_storage storage{};
  socklen_t length = sizeof(storage);
  errno = 0;
  const int value =
      ::getsockname(fd, reinterpret_cast<sockaddr *>(&storage), &length);
  return PosixAddressResult(
      value < 0 ? -1 : 0, value < 0 ? errno : 0,
      value < 0 ? ::rund::net::Address{}
                : AddressFromPosix(reinterpret_cast<const sockaddr *>(&storage),
                                   length));
}

NativeCallResult NativeShutdown(const int fd,
                                const ::rund::net::ShutdownMode mode) noexcept {
  int how = 0;
  switch (mode) {
  case ::rund::net::ShutdownMode::Read:
    how = SHUT_RD;
    break;
  case ::rund::net::ShutdownMode::Write:
    how = SHUT_WR;
    break;
  case ::rund::net::ShutdownMode::ReadWrite:
    how = SHUT_RDWR;
    break;
  }
  errno = 0;
  const int value = ::shutdown(fd, how);
  return PosixNetResult(value < 0 ? -1 : 0, value < 0 ? errno : 0);
}

NativeCallResult NativePrepareSocketSend(const int fd) noexcept {
  return posix_net::PrepareSocketSend(fd);
}

NativeCallResult NativeRecv(const int fd,
                            const std::span<std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return PosixNetResult(-1, EINVAL, true);
  }
  if (buffer.empty()) {
    return PosixNetResult(0, 0);
  }
  errno = 0;
  const ssize_t value =
      ::recv(fd, static_cast<void *>(buffer.data()), buffer.size(), 0);
  return PosixNetResult(value, value < 0 ? errno : 0);
}

NativeCallResult NativeSend(const int fd,
                            const std::span<const std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return PosixNetResult(-1, EINVAL, true);
  }
  if (buffer.empty()) {
    return PosixNetResult(0, 0);
  }
  errno = 0;
  const ssize_t value = ::send(fd, static_cast<const void *>(buffer.data()),
                               buffer.size(), posix_net::DirectSendFlags());
  return PosixNetResult(value, value < 0 ? errno : 0);
}

NativeCallResult NativeTryRecv(const int fd,
                               const std::span<std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return PosixNetResult(-1, EINVAL, true);
  }
  if (buffer.empty()) {
    return PosixNetResult(0, 0);
  }
  errno = 0;
  const ssize_t value = ::recv(fd, static_cast<void *>(buffer.data()),
                               buffer.size(), posix_net::TryRecvFlags());
  return PosixNetResult(value, value < 0 ? errno : 0);
}

NativeCallResult
NativeTrySend(const int fd, const std::span<const std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return PosixNetResult(-1, EINVAL, true);
  }
  if (buffer.empty()) {
    return PosixNetResult(0, 0);
  }
  errno = 0;
  const ssize_t value = ::send(fd, static_cast<const void *>(buffer.data()),
                               buffer.size(), posix_net::TrySendFlags());
  return PosixNetResult(value, value < 0 ? errno : 0);
}

NativeAddressResult NativeAccept(const int fd) noexcept {
  sockaddr_storage storage{};
  socklen_t length = sizeof(storage);
  errno = 0;
  const int value =
      ::accept(fd, reinterpret_cast<sockaddr *>(&storage), &length);
  return PosixAddressResult(
      value, value < 0 ? errno : 0,
      value < 0 ? ::rund::net::Address{}
                : AddressFromPosix(reinterpret_cast<const sockaddr *>(&storage),
                                   length));
}

NativeCallResult NativeConnect(const int fd,
                               const ::rund::net::Address &address) noexcept {
  sockaddr_storage storage{};
  socklen_t length = 0;
  if (!PosixAddress(address, &storage, &length)) {
    return PosixNetResult(-1, EINVAL, true);
  }
  errno = 0;
  const int value =
      ::connect(fd, reinterpret_cast<const sockaddr *>(&storage), length);
  return PosixNetResult(value, value < 0 ? errno : 0);
}

} // namespace rund::node
