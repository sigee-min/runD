#pragma once

#include <rund/net/accept.hpp>
#include <rund/net/handoff.hpp>
#include <rund/net/socket.hpp>

#include <cstdint>

namespace rund::net::server {

struct Options {
  net::SocketView listener{};
  net::accept::Budget accepts{};
  net::accept::Options accepted{};
  const char *task_name = "net.peer";
};

} // namespace rund::net::server
