#pragma once

#include <rund/net/socket.hpp>

namespace rund::net::server {

struct Peer {
  Socket socket{};
};

} // namespace rund::net::server
