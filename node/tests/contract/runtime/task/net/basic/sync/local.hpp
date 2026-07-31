#pragma once

#include "../local.hpp"
#include <rund/net/socket.hpp>

struct BasicSyncSockets {
  rund::net::Socket left{};
  rund::net::Socket right{};
};

[[nodiscard]] bool OpenBasicSyncSockets(BasicSyncSockets &sockets);
[[nodiscard]] int RunNetBasicSyncTransferCase();
[[nodiscard]] int RunNetBasicSyncZeroCase();
[[nodiscard]] int RunNetBasicSyncInvalidCase();
