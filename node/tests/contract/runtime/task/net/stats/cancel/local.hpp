#pragma once

#include "../local.hpp"
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/status.hpp>

struct NetStatsCancelCase {
  bool cancel_pair_ok = false;
  bool close_pair_ok = false;
  bool setup_ok = false;
  bool source_valid = false;
  bool token_valid = false;
  bool cancel_ok = false;
  rund::Session::Result report{};
  rund::net::ready::Ticket cancel_result{};
  rund::net::ready::Ticket close_wait_result{};
  rund::net::CloseResult close_result{};
  rund::task::Status cancel_yielded{};
  rund::task::Status close_yielded{};
  rund::task::Status joined{};
};

[[nodiscard]] NetStatsCancelCase RunNetStatsCancelScenario();
[[nodiscard]] bool NetStatsCancelResultMatches(const NetStatsCancelCase &stats);
[[nodiscard]] bool NetStatsCloseResultMatches(const NetStatsCancelCase &stats);
[[nodiscard]] bool NetStatsCancelCountersMatch(const NetStatsCancelCase &stats);
