#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <chrono>
#include <span>

ReadyManyBoundaryCase RunReadyManyBoundaryScenario() {
  ReadyManyBoundaryCase boundary{};
  SocketPairCleanup cleanup{};
  boundary.setup_ok = MakeSocketPair(cleanup);
  if (!boundary.setup_ok) {
    return boundary;
  }

  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  boundary.nonblocking = rund::net::nonblocking(reader.view(), true);
  const std::array<rund::net::ready::Request, 1u> request{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
  };
  const std::array<rund::net::ready::Request, 2u> duplicate_requests{
      request[0u],
      request[0u],
  };
  const std::array<rund::net::ready::Request, 1u> invalid_request{
      rund::net::ready::Request{.socket = rund::net::SocketView{},
                                .interest =
                                    rund::net::ready::Interest::Readable},
  };
  std::array<rund::net::ready::Event, 2u> events{};

  boundary.report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      boundary.empty = rund::net::ready::many::wait(
                           std::span<const rund::net::ready::Request>{},
                           std::span<rund::net::ready::Event>{})
                           .wait();
      boundary.empty_output =
          rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{request},
              std::span<rund::net::ready::Event>{})
              .wait();
      boundary.zero_budget =
          rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{request},
              std::span<rund::net::ready::Event>{events},
              rund::net::ready::many::Budget{.max_events = 0u})
              .wait();
      boundary.invalid =
          rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{invalid_request},
              std::span<rund::net::ready::Event>{events})
              .wait();
      boundary.duplicate = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{duplicate_requests},
          std::span<rund::net::ready::Event>{events});
      boundary.negative_timeout =
          rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{request},
              std::span<rund::net::ready::Event>{events},
              std::chrono::milliseconds{-1})
              .wait();
      boundary.zero_timeout = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{request},
          std::span<rund::net::ready::Event>{events},
          std::chrono::nanoseconds{0});
      boundary.parked_timeout = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{request},
          std::span<rund::net::ready::Event>{events},
          std::chrono::milliseconds{1});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-boundaries", wait());
    boundary.joined = rund::task::join(waiter);
  });
  return boundary;
}
