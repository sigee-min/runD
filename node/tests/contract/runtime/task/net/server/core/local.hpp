#pragma once

#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <cstddef>

#include <netinet/in.h>

struct ServerSocketCleanup {
  int fd = -1;

  ~ServerSocketCleanup();

  ServerSocketCleanup() = default;
  explicit ServerSocketCleanup(int native) noexcept;
  ServerSocketCleanup(const ServerSocketCleanup &) = delete;
  ServerSocketCleanup &operator=(const ServerSocketCleanup &) = delete;

  void reset(int native) noexcept;
};

struct LoopbackFixture {
  ServerSocketCleanup listener_cleanup{};
  sockaddr_in address{};
  rund::net::Socket listener{};
};

[[nodiscard]] int PrepareLoopbackListener(LoopbackFixture &fixture);
[[nodiscard]] int StartLoopbackClient(const sockaddr_in &address,
                                      ServerSocketCleanup &cleanup);
[[nodiscard]] int StartLoopbackClientWithByte(const sockaddr_in &address,
                                              std::byte byte,
                                              ServerSocketCleanup &cleanup);
[[nodiscard]] rund::SessionConfig NetServerRunSpec() noexcept;

[[nodiscard]] int RunServerInvalidListenerCase();
[[nodiscard]] int RunServerInlineLoopbackCase();
