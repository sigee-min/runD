#include "../local.hpp"

#include <cstring>
#include <rund/net/address.hpp>

namespace rund::node::test_contract::net_accept_handoff {

rund::net::Address AddressFromSockaddr(const sockaddr_in &address) {
  std::array<std::byte, 4u> bytes{};
  std::memcpy(bytes.data(), &address.sin_addr, bytes.size());
  return rund::net::Address::ipv4(bytes, ntohs(address.sin_port));
}

} // namespace rund::node::test_contract::net_accept_handoff
