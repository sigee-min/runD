#include "../local.hpp"

#include <cstring>
#include <rund/net/address.hpp>

namespace rund::node::test_contract::net_limits {

rund::net::Address LoopbackAnyPort() {
  return rund::net::Address::loopback(rund::net::Family::IPv4);
}

} // namespace rund::node::test_contract::net_limits
