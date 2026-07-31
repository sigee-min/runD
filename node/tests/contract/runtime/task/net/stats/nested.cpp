#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "src/runtime/task/scheduler/host.hpp"
#include "test/assert.hpp"

#include <rund/replay.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include <sys/socket.h>

namespace {

void EmitSyntheticNetworkEvents(const bool overflow, bool &committed) {
  const auto record = [&](const rund::host::EventKind kind,
                          const rund::host::Status status,
                          const std::uint64_t completed_bytes) {
    committed = rund::node::scheduler_host::Record(rund::host::Event{
                    .kind = kind,
                    .status = status,
                    .completed_bytes = completed_bytes,
                }) &&
                committed;
  };
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  if (overflow) {
    record(rund::host::EventKind::NetRecv, rund::host::Status::Ok,
           maximum - 1u);
    record(rund::host::EventKind::NetRecv, rund::host::Status::Ok, 2u);
    record(rund::host::EventKind::NetSend, rund::host::Status::Ok, maximum);
    record(rund::host::EventKind::NetSend, rund::host::Status::Ok, 1u);
    return;
  }
  record(rund::host::EventKind::NetRecv, rund::host::Status::Ok, 7u);
  record(rund::host::EventKind::NetRecv, rund::host::Status::Ok, 0u);
  record(rund::host::EventKind::NetRecv, rund::host::Status::WouldBlock,
         maximum);
  record(rund::host::EventKind::NetRecv, rund::host::Status::SyscallFailed,
         maximum);
  record(rund::host::EventKind::NetSend, rund::host::Status::Ok, 11u);
  record(rund::host::EventKind::NetSend, rund::host::Status::Ok, 0u);
  record(rund::host::EventKind::NetSend, rund::host::Status::WouldBlock,
         maximum);
  record(rund::host::EventKind::NetSend, rund::host::Status::SyscallFailed,
         maximum);
}

} // namespace

int NetStatsNestedVisibility() {
  NetStatsSocketPairCleanup cleanup{};
  TEST_ASSERT(MakeNetStatsSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket writer = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true).ok());

  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
  };
  std::array<rund::net::ready::Event, 1u> events{};
  std::array<std::byte, 1u> byte{std::byte{'s'}};
  rund::net::ready::many::Result ready_result{};
  rund::net::SendResult send_result{};
  rund::task::Status send_yielded{};
  rund::task::Status joined{};
  rund::net::CloseResult listener_closed{};

  const rund::Session::Result report = rund::run(NetStatsRunSpec(), [&] {
    auto opened = rund::net::open(
        rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                               .transport = rund::net::Transport::Stream,
                               .nonblocking = true});
    TEST_ASSERT(opened.ok());
    NetStatsSocketCloseGuard listener_guard{std::move(opened.socket)};
    const auto bound = rund::net::bind(listener_guard.socket.view(),
                                       NetStatsLoopbackAnyPort());
    TEST_ASSERT(bound.ok());
    const auto listened = rund::net::listen(listener_guard.socket.view(), 8);
    TEST_ASSERT(listened.ok());

    auto wait = [&]() -> rund::task::Task<void> {
      ready_result = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events}, std::chrono::seconds{30});
    };
    auto send = [&]() -> rund::task::Task<void> {
      send_yielded = co_await rund::task::yield();
      send_result = rund::net::send(
          rund::node::test::net::ticket(writer.view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-stats-ready-many-waiter", wait());
    const rund::task::Handle sender =
        rund::task::spawn("net-stats-ready-many-sender", send());
    joined = rund::task::join(waiter, sender);
    listener_closed = listener_guard.close();
    return 0;
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(send_yielded.ok());
  TEST_ASSERT(send_result.ok());
  TEST_ASSERT(ready_result.ok());
  TEST_ASSERT(ready_result.events == 1u);
  TEST_ASSERT(listener_closed.ok());
  TEST_ASSERT(report.tasks().network().sockets_opened() == 1u);
  TEST_ASSERT(report.tasks().network().sockets_bound() == 1u);
  TEST_ASSERT(report.tasks().network().sockets_listened() == 1u);
  TEST_ASSERT(report.tasks().reactor().ready_many_requests() == 1u);
  TEST_ASSERT(report.tasks().reactor().ready_many_events() >= 1u);
  TEST_ASSERT(report.tasks().reactor().registration_apply_calls() >= 1u);
  return 0;
}

int NetStatsByteAccounting() {
  rund::Session session{};
  TEST_ASSERT(session.open(rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .host_event_capacity = 16u,
          },
  }));
  bool committed = true;
  auto regular = [&](rund::replay::Context &) {
    EmitSyntheticNetworkEvents(false, committed);
  };
  const rund::replay::Record recorded = rund::replay::record(session, regular);
  TEST_ASSERT(recorded.ok());
  TEST_ASSERT(committed);
  TEST_ASSERT(recorded.host_event_count() == 8u);
  TEST_ASSERT(recorded.tasks().network().recv_calls() == 2u);
  TEST_ASSERT(recorded.tasks().network().send_calls() == 2u);
  TEST_ASSERT(recorded.tasks().network().would_block() == 2u);
  TEST_ASSERT(recorded.tasks().network().bytes_received() == 7u);
  TEST_ASSERT(recorded.tasks().network().bytes_sent() == 11u);

  committed = true;
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, regular);
  TEST_ASSERT(replayed.ok());
  TEST_ASSERT(committed);
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(replayed.actual()->tasks().network().bytes_received() ==
              recorded.tasks().network().bytes_received());
  TEST_ASSERT(replayed.actual()->tasks().network().bytes_sent() ==
              recorded.tasks().network().bytes_sent());
  TEST_ASSERT(replayed.actual()->tasks().trace_hash() ==
              recorded.tasks().trace_hash());

  committed = true;
  const rund::replay::Record saturated =
      rund::replay::record(session, [&](rund::replay::Context &) {
        EmitSyntheticNetworkEvents(true, committed);
      });
  TEST_ASSERT(saturated.ok());
  TEST_ASSERT(committed);
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  TEST_ASSERT(saturated.tasks().network().bytes_received() == maximum);
  TEST_ASSERT(saturated.tasks().network().bytes_sent() == maximum);
  TEST_ASSERT(session.close());
  return 0;
}
