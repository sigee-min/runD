#pragma once

#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

struct NetStatsSocketPairCleanup {
  int left = -1;
  int right = -1;

  ~NetStatsSocketPairCleanup();

  NetStatsSocketPairCleanup() = default;
  NetStatsSocketPairCleanup(const NetStatsSocketPairCleanup &) = delete;
  NetStatsSocketPairCleanup &
  operator=(const NetStatsSocketPairCleanup &) = delete;
};

struct NetStatsSocketCloseGuard {
  rund::net::Socket socket{};

  NetStatsSocketCloseGuard() = default;
  explicit NetStatsSocketCloseGuard(rund::net::Socket opened) noexcept;
  NetStatsSocketCloseGuard(const NetStatsSocketCloseGuard &) = delete;
  NetStatsSocketCloseGuard &
  operator=(const NetStatsSocketCloseGuard &) = delete;

  NetStatsSocketCloseGuard(NetStatsSocketCloseGuard &&) noexcept = default;
  NetStatsSocketCloseGuard &
  operator=(NetStatsSocketCloseGuard &&) noexcept = default;

  [[nodiscard]] rund::net::CloseResult close() noexcept;
};

[[nodiscard]] bool MakeNetStatsSocketPair(NetStatsSocketPairCleanup &cleanup);
[[nodiscard]] rund::SessionConfig NetStatsRunSpec() noexcept;
[[nodiscard]] rund::net::Address NetStatsLoopbackAnyPort();

[[nodiscard]] int NetStatsNestedVisibility();
[[nodiscard]] int NetStatsByteAccounting();
[[nodiscard]] int NetStatsCloseTimeoutCancellation();
