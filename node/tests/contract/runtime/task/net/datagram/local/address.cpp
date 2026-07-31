#include "../local.hpp"

#include <algorithm>
#include <rund/net/address.hpp>

[[nodiscard]] rund::net::Address LoopbackAnyPort() {
  return rund::net::Address::loopback(rund::net::Family::IPv4);
}

[[nodiscard]] bool IsLoopbackPort(const rund::net::Address address) noexcept {
  constexpr std::array<std::byte, 4u> kLoopback{std::byte{127u}, std::byte{0u},
                                                std::byte{0u}, std::byte{1u}};
  return address.family() == rund::net::Family::IPv4 &&
         std::ranges::equal(address.bytes(), kLoopback) && address.port() != 0u;
}
