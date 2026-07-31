#pragma once

#include "../local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/status.hpp>

struct ReadyManyBoundaryCase {
  bool setup_ok = false;
  rund::net::NonblockingResult nonblocking{};
  rund::Session::Result report{};
  rund::task::Status joined{};
  rund::net::ready::many::Result empty{};
  rund::net::ready::many::Result empty_output{};
  rund::net::ready::many::Result zero_budget{};
  rund::net::ready::many::Result invalid{};
  rund::net::ready::many::Result duplicate{};
  rund::net::ready::many::Result negative_timeout{};
  rund::net::ready::many::Result zero_timeout{};
  rund::net::ready::many::Result parked_timeout{};
};

[[nodiscard]] ReadyManyBoundaryCase RunReadyManyBoundaryScenario();
[[nodiscard]] bool
ReadyManyBoundaryEmptyMatches(const ReadyManyBoundaryCase &boundary);
[[nodiscard]] bool
ReadyManyBoundaryBudgetMatches(const ReadyManyBoundaryCase &boundary);
[[nodiscard]] bool
ReadyManyBoundaryRejectMatches(const ReadyManyBoundaryCase &boundary);
[[nodiscard]] bool
ReadyManyBoundaryTimeoutMatches(const ReadyManyBoundaryCase &boundary);
