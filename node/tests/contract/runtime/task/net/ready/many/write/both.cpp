#include "src/host/net/test/socket.hpp"
#include "src/runtime/reactor/readiness/state.hpp"
#include "test/assert.hpp"

#include "../local.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <span>
#include <utility>

#include <sys/socket.h>

namespace {

[[nodiscard]] bool FillSendBuffer(const int native,
                                  std::size_t &filled) noexcept {
  std::array<std::byte, 16u * 1024u> payload{};
  filled = 0u;
  for (;;) {
    const ssize_t sent = ::send(native, payload.data(), payload.size(), 0);
    if (sent > 0) {
      filled += static_cast<std::size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    return sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
           filled != 0u;
  }
}

[[nodiscard]] bool DrainBytes(const int native,
                              const std::size_t expected) noexcept {
  std::array<std::byte, 16u * 1024u> payload{};
  std::size_t drained = 0u;
  while (drained < expected) {
    const std::size_t remaining = expected - drained;
    const ssize_t received =
        ::recv(native, payload.data(), std::min(remaining, payload.size()), 0);
    if (received > 0) {
      drained += static_cast<std::size_t>(received);
      continue;
    }
    if (received < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

[[nodiscard]] bool SendByte(const int native) noexcept {
  const std::byte byte{'p'};
  for (;;) {
    const ssize_t sent = ::send(native, &byte, sizeof(byte), 0);
    if (sent == static_cast<ssize_t>(sizeof(byte))) {
      return true;
    }
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
}

void VerifyParkedReadWriteCoalescing() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket target = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket peer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(target.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(peer.view(), true).ok());

  const int target_native = rund::node::test::net::native(target);
  const int peer_native = rund::node::test::net::native(peer);
  std::size_t filled = 0u;
  TEST_ASSERT(FillSendBuffer(target_native, filled));

  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = target.view(),
                                .interest =
                                    rund::net::ready::Interest::ReadWrite}};
  std::array<rund::net::ready::Event, 1u> events{};
  rund::net::ready::many::Result result{};
  rund::task::Status trigger_yield{};
  rund::task::Status joined{};
  bool drained = false;
  bool sent = false;
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events},
          std::chrono::seconds{5});
    };
    auto trigger = [&]() -> rund::task::Task<void> {
      trigger_yield = co_await rund::task::yield();
      drained = DrainBytes(peer_native, filled);
      sent = SendByte(peer_native);
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-write-both-parked", wait());
    const rund::task::Handle trigger_task =
        rund::task::spawn("net-ready-many-write-both-trigger", trigger());
    joined = rund::task::join(waiter, trigger_task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(trigger_yield.ok());
  TEST_ASSERT(drained);
  TEST_ASSERT(sent);
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 1u);
  TEST_ASSERT(events[0u].index == 0u);
  TEST_ASSERT(events[0u].ticket.ready());
  TEST_ASSERT(events[0u].ticket.id() == target.id());
  TEST_ASSERT(events[0u].ticket.interest() ==
              rund::net::ready::Interest::ReadWrite);
  const short revents = events[0u].ticket.revents();
  TEST_ASSERT((revents & static_cast<short>(rund::node::ReactorEvent::Read)) !=
              0);
  TEST_ASSERT((revents & static_cast<short>(rund::node::ReactorEvent::Write)) !=
              0);
}

} // namespace

int RunNetReadyManyWriteReadWriteCase() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());
  std::array<std::byte, 1u> payload{std::byte{'r'}};
  TEST_ASSERT(rund::net::direct::send(writer.view(),
                                      std::span<const std::byte>{payload})
                  .ok());
  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::ReadWrite}};
  std::array<rund::net::ready::Event, 1u> events{};
  rund::net::ready::many::Result result{};
  std::array<std::byte, 1u> received{};
  rund::net::ReceiveResult receive{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(ReadyManyRunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events});
      if (result.ok() && result.events == 1u) {
        receive = rund::net::receive(std::move(events[0u].ticket), received);
      }
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-many-write-both", wait());
    joined = rund::task::join(waiter);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(result.ok());
  TEST_ASSERT(result.events == 1u);
  TEST_ASSERT(events[0u].index == 0u);
  TEST_ASSERT(events[0u].ticket.id() == reader.id());
  TEST_ASSERT(events[0u].ticket.interest() ==
              rund::net::ready::Interest::ReadWrite);
  TEST_ASSERT(events[0u].ticket.consumed());
  TEST_ASSERT(receive);
  TEST_ASSERT(receive.bytes == 1);
  TEST_ASSERT(received == payload);
  VerifyParkedReadWriteCoalescing();
  return 0;
}
