#pragma once

#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <cstddef>

#include <netinet/in.h>

struct MultiClientSocketCleanup {
  int fd = -1;

  ~MultiClientSocketCleanup();

  MultiClientSocketCleanup() = default;
  explicit MultiClientSocketCleanup(int native) noexcept;
  MultiClientSocketCleanup(const MultiClientSocketCleanup &) = delete;
  MultiClientSocketCleanup &
  operator=(const MultiClientSocketCleanup &) = delete;
  MultiClientSocketCleanup(MultiClientSocketCleanup &&other) noexcept;
  MultiClientSocketCleanup &
  operator=(MultiClientSocketCleanup &&other) noexcept;

  void reset(int native) noexcept;
};

struct MultiClientLoopbackFixture {
  MultiClientSocketCleanup listener_cleanup{};
  rund::net::Socket listener{};
  sockaddr_in connect_address{};
};

[[nodiscard]] int
PrepareMultiClientLoopbackListener(MultiClientLoopbackFixture &fixture);
[[nodiscard]] int
StartMultiClientBlockingClient(const sockaddr_in &connect_address,
                               std::byte byte,
                               MultiClientSocketCleanup &client_cleanup);
[[nodiscard]] bool RecvMultiClientOneWithTimeout(int fd, std::byte &out);
[[nodiscard]] rund::SessionConfig NetMultiClientRunSpec() noexcept;

[[nodiscard]] int RunNetMultiClientSubstrateCase();
