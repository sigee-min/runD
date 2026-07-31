#pragma once

#include "../local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>

struct AcceptConnectEventCase {
  bool listener_ok = false;
  bool client_ok = false;
  rund::net::Socket listener{};
  rund::net::Socket client{};
  rund::net::NonblockingResult listener_nonblocking{};
  rund::net::NonblockingResult client_nonblocking{};
  rund::net::Address address{};
  rund::net::accept::Result empty_accept{};
  rund::net::accept::Result accepted{};
  rund::net::connect::Result started{};
  rund::net::connect::Result finished{};
  rund::Session::Result report{};
};

[[nodiscard]] AcceptConnectEventCase RunAcceptConnectEventScenario();
[[nodiscard]] bool
AcceptConnectAcceptEvidenceMatches(const AcceptConnectEventCase &event);
[[nodiscard]] bool
AcceptConnectConnectEvidenceMatches(const AcceptConnectEventCase &event);
