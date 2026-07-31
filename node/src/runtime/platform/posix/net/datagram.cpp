#include <sys/socket.h>

#include <cerrno>

#include "../../net.hpp"
#include "../address.hpp"
#include "../buffer.hpp"
#include "result.hpp"
#include "socket/call.hpp"

namespace rund::node {

NativeAddressResult NativeRecvFrom(const int fd,
                                   const std::span<std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return NativeAddressResult{
        .call = PosixNetResult(-1, EINVAL, true),
    };
  }
  sockaddr_storage storage{};
  socklen_t length = sizeof(storage);
  errno = 0;
  const ssize_t value =
      ::recvfrom(fd, static_cast<void *>(buffer.data()), buffer.size(),
                 posix_net::TryRecvFlags(),
                 reinterpret_cast<sockaddr *>(&storage), &length);
  return PosixAddressResult(
      value, value < 0 ? errno : 0,
      value < 0 ? ::rund::net::Address{}
                : AddressFromPosix(reinterpret_cast<const sockaddr *>(&storage),
                                   length));
}

NativeCallResult NativeSendTo(const int fd,
                              const std::span<const std::byte> buffer,
                              const ::rund::net::Address &address) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return PosixNetResult(-1, EINVAL, true);
  }
  sockaddr_storage storage{};
  socklen_t length = 0;
  if (!PosixAddress(address, &storage, &length)) {
    return PosixNetResult(-1, EINVAL, true);
  }
  errno = 0;
  const ssize_t value =
      ::sendto(fd, static_cast<const void *>(buffer.data()), buffer.size(),
               posix_net::TrySendFlags(),
               reinterpret_cast<const sockaddr *>(&storage), length);
  return PosixNetResult(value, value < 0 ? errno : 0);
}

} // namespace rund::node
