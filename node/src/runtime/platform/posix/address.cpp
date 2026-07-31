#include "address.hpp"

#include <netinet/in.h>

#include <array>
#include <cstring>

namespace rund::node {

bool PosixAddress(const ::rund::net::Address &address,
                  sockaddr_storage *const storage,
                  socklen_t *const length) noexcept {
  if (!address || storage == nullptr || length == nullptr) {
    return false;
  }

  std::memset(storage, 0, sizeof(*storage));
  if (address.family() == ::rund::net::Family::IPv4) {
    sockaddr_in native{};
    native.sin_family = AF_INET;
    native.sin_port = htons(address.port());
    std::memcpy(&native.sin_addr, address.bytes().data(),
                address.bytes().size());
    std::memcpy(storage, &native, sizeof(native));
    *length = sizeof(native);
    return true;
  }
  if (address.family() == ::rund::net::Family::IPv6) {
    sockaddr_in6 native{};
    native.sin6_family = AF_INET6;
    native.sin6_port = htons(address.port());
    std::memcpy(&native.sin6_addr, address.bytes().data(),
                address.bytes().size());
    std::memcpy(storage, &native, sizeof(native));
    *length = sizeof(native);
    return true;
  }
  return false;
}

::rund::net::Address AddressFromPosix(const sockaddr *const address,
                              const socklen_t length) noexcept {
  if (address == nullptr) {
    return {};
  }
  if (address->sa_family == AF_INET && length == sizeof(sockaddr_in)) {
    const auto *const native = reinterpret_cast<const sockaddr_in *>(address);
    std::array<std::byte, 4u> bytes{};
    std::memcpy(bytes.data(), &native->sin_addr, bytes.size());
    return ::rund::net::Address::ipv4(bytes, ntohs(native->sin_port));
  }
  if (address->sa_family == AF_INET6 && length == sizeof(sockaddr_in6)) {
    const auto *const native = reinterpret_cast<const sockaddr_in6 *>(address);
    std::array<std::byte, 16u> bytes{};
    std::memcpy(bytes.data(), &native->sin6_addr, bytes.size());
    return ::rund::net::Address::ipv6(bytes, ntohs(native->sin6_port));
  }
  return {};
}

} // namespace rund::node
