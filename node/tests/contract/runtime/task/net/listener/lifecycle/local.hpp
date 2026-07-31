#pragma once

#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>

[[nodiscard]] rund::net::Address ListenerLoopbackAnyPort();
[[nodiscard]] rund::net::Address ListenerLoopbackV6AnyPort();

struct ListenerSocketCloseGuard {
  rund::net::Socket socket{};

  ListenerSocketCloseGuard() = default;
  explicit ListenerSocketCloseGuard(rund::net::Socket opened) noexcept;
  ListenerSocketCloseGuard(const ListenerSocketCloseGuard &) = delete;
  ListenerSocketCloseGuard &
  operator=(const ListenerSocketCloseGuard &) = delete;

  ListenerSocketCloseGuard(ListenerSocketCloseGuard &&) noexcept = default;
  ListenerSocketCloseGuard &
  operator=(ListenerSocketCloseGuard &&) noexcept = default;

  [[nodiscard]] rund::net::CloseResult close() noexcept;
};

[[nodiscard]] int ListenerLifecycleOpensBindsListensAndCloses();
[[nodiscard]] int ListenerInvalidInputsFailClosed();
[[nodiscard]] int ListenerBacklogAndStaleHandlesFailClosed();
[[nodiscard]] int ListenerShutdownConnectedSocketSucceeds();
