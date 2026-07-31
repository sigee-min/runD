#pragma once

#include <rund/net/address.hpp>
#include <rund/net/options.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <iostream>

#define NET_OPTIONS_ASSERT(condition)                                          \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "NET_OPTIONS_ASSERT failed: " #condition " at " << __FILE__ \
                << ':' << __LINE__ << '\n';                                    \
      return false;                                                            \
    }                                                                          \
  } while (false)

struct NetOptionsSocketCloseGuard {
  rund::net::Socket socket{};

  NetOptionsSocketCloseGuard() = default;
  explicit NetOptionsSocketCloseGuard(rund::net::Socket opened) noexcept;
  NetOptionsSocketCloseGuard(const NetOptionsSocketCloseGuard &) = delete;
  NetOptionsSocketCloseGuard &
  operator=(const NetOptionsSocketCloseGuard &) = delete;
  NetOptionsSocketCloseGuard(NetOptionsSocketCloseGuard &&) noexcept = default;
  NetOptionsSocketCloseGuard &
  operator=(NetOptionsSocketCloseGuard &&) noexcept = default;
};

[[nodiscard]] bool OpenNetOptionsInetSocket(rund::net::Transport transport,
                                            NetOptionsSocketCloseGuard &guard);
[[nodiscard]] bool
IsTcpNoDelayUdpAcceptedFailure(rund::net::option::Result result) noexcept;

[[nodiscard]] bool NetSocketOptionsSetAndReadSelectedOptions();
[[nodiscard]] bool NetSocketOptionsRejectInvalidValues();
[[nodiscard]] bool NetSocketOptionsRejectInvalidOptionIdsBeforeEvents();
[[nodiscard]] bool NetSocketOptionsEventsReplayStable();
