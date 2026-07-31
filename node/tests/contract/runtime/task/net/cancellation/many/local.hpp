#pragma once

#include "../local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/status.hpp>

struct NetCancellationManyCase {
  bool left_pair_ok = false;
  bool right_pair_ok = false;
  bool setup_ok = false;
  bool source_valid = false;
  bool token_valid = false;
  bool cancel_ok = false;
  rund::Session::Result report{};
  rund::net::ready::many::Result ready{};
  rund::task::Status cancel_yield{};
  rund::net::CloseResult close{};
  rund::task::Status scope{};
  rund::task::Status post_close_scope{};
  rund::task::Status post_close_sleep{};
};

[[nodiscard]] NetCancellationManyCase RunNetCancellationManyScenario();
[[nodiscard]] bool
NetCancellationManyCancelMatches(const NetCancellationManyCase &many);
[[nodiscard]] bool
NetCancellationManyCleanupMatches(const NetCancellationManyCase &many);
[[nodiscard]] bool
NetCancellationManyCountersMatch(const NetCancellationManyCase &many);
