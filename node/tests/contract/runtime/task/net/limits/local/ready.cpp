#include "../local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>

namespace rund::node::test_contract::net_limits {

rund::net::ready::Request
ReadableRequest(const rund::net::SocketView socket) noexcept {
  return rund::net::ready::Request{
      .socket = socket, .interest = rund::net::ready::Interest::Readable};
}

} // namespace rund::node::test_contract::net_limits
