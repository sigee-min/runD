#pragma once

#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <span>

struct AcceptDrainSocketCleanup {
  rund::net::Socket socket{};
  int fd = -1;

  ~AcceptDrainSocketCleanup();

  AcceptDrainSocketCleanup() = default;
  explicit AcceptDrainSocketCleanup(int native) noexcept;
  AcceptDrainSocketCleanup(const AcceptDrainSocketCleanup &) = delete;
  AcceptDrainSocketCleanup &
  operator=(const AcceptDrainSocketCleanup &) = delete;
  AcceptDrainSocketCleanup(AcceptDrainSocketCleanup &&other) noexcept;
  AcceptDrainSocketCleanup &
  operator=(AcceptDrainSocketCleanup &&other) noexcept;
};

struct AcceptDrainLoopbackFixture {
  AcceptDrainSocketCleanup listener_cleanup{};
  rund::net::Socket listener{};
  rund::net::Address connect_address{};
};

[[nodiscard]] rund::SessionConfig AcceptDrainRunSpec() noexcept;
[[nodiscard]] int
PrepareAcceptDrainLoopbackListener(AcceptDrainLoopbackFixture &fixture);
[[nodiscard]] int
StartAcceptDrainClients(const rund::net::Address &connect_address,
                        std::span<AcceptDrainSocketCleanup> clients);

[[nodiscard]] int RunAcceptDrainWouldBlockCase();
[[nodiscard]] int RunAcceptDrainCallbackStopCase();
[[nodiscard]] int RunAcceptDrainBudgetExhaustedCase();
[[nodiscard]] int RunAcceptDrainNonWouldBlockFailureCase();
