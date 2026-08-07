#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/result.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/net/server/task.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace {

struct SynchronousThrow final {
  rund::net::CloseResult *closed = nullptr;

  [[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
  operator()(rund::net::server::Peer peer) const {
    *closed = peer.socket.close();
    throw 1;
  }
};

struct PublicClient final {
  rund::net::connect::Result started{};
  rund::net::connect::Result connected{};
  rund::net::SendResult sent{};
};

[[nodiscard]] rund::task::Task<void>
SendPublicByte(const rund::net::SocketView socket,
               const rund::net::Address address,
               const std::span<const std::byte> bytes,
               PublicClient &result) {
  result.started = rund::net::connect::start(socket, address);
  if (!result.started) {
    co_return;
  }
  auto writable = co_await rund::net::ready::write(socket);
  if (!writable.ready()) {
    result.connected = rund::net::connect::Result{writable.code()};
    co_return;
  }
  result.connected =
      rund::net::connect::finish(std::move(writable), address);
  if (!result.connected) {
    co_return;
  }
  result.sent = co_await rund::net::send(socket, bytes);
}

[[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
ReceivePublicByte(rund::net::server::Peer peer, std::byte &value) {
  std::array<std::byte, 1u> bytes{};
  const rund::net::ReceiveResult received =
      co_await rund::net::receive(peer.socket.view(), bytes);
  if (!received || received.bytes != 1) {
    co_return rund::net::server::PeerResult::fail(
        received ? rund::ReasonCode::NetPeerHandlerFailed : received.code());
  }
  value = bytes[0];
  co_return rund::net::server::PeerResult::complete();
}

[[nodiscard]] bool HasCounts(const rund::net::server::Result &result,
                             const std::uint32_t completed,
                             const std::uint32_t failed,
                             const std::uint32_t stopped) noexcept {
  return result.accepted == 1u && result.started == 1u &&
         result.completed == completed && result.failed == failed &&
         result.stopped == stopped && result.rejected == 0u;
}

[[nodiscard]] std::uint64_t
CountEvents(const rund::Session::Result &run,
            const rund::host::EventKind kind) noexcept {
  std::uint64_t count = 0u;
  for (const rund::host::Event &event : run.events()) {
    count += event.kind == kind ? 1u : 0u;
  }
  return count;
}

[[nodiscard]] int RunServerInlineOutcomeCase() {
  constexpr std::size_t kClients = 4u;
  LoopbackFixture fixture{};
  TEST_ASSERT(PrepareLoopbackListener(fixture) == 0);
  std::array<ServerSocketCleanup, kClients> clients{};
  for (ServerSocketCleanup &client : clients) {
    TEST_ASSERT(StartLoopbackClient(fixture.address, client) == 0);
  }

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 1u;
  rund::net::server::Result stopped{};
  rund::net::server::Result task_failed{};
  rund::net::server::Result invocation_failed{};
  rund::net::server::Result flattened_failed{};
  rund::net::CloseResult stopped_close{};
  rund::net::CloseResult task_failed_close{};
  rund::net::CloseResult invocation_failed_close{};
  rund::net::CloseResult flattened_failed_close{};
  rund::task::Status joined{};

  const rund::Session::Result run = rund::run(NetServerRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      stopped = co_await rund::net::server::serve(
          options,
          [&](rund::net::server::Peer peer)
              -> rund::task::Task<rund::net::server::PeerResult> {
            stopped_close = peer.socket.close();
            co_return rund::net::server::PeerResult::stop();
          });

      task_failed = co_await rund::net::server::serve(
          options,
          [&](rund::net::server::Peer peer)
              -> rund::task::Task<rund::net::server::PeerResult> {
            task_failed_close = peer.socket.close();
            throw 1;
            co_return rund::net::server::PeerResult::complete();
          });

      invocation_failed = co_await rund::net::server::serve(
          options, SynchronousThrow{.closed = &invocation_failed_close});

      flattened_failed = co_await rund::net::server::serve(
          options,
          [&](rund::net::server::Peer peer)
              -> rund::task::Task<rund::net::server::PeerResult> {
            flattened_failed_close = peer.socket.close();
            return rund::task::Task<rund::net::server::PeerResult>{
                rund::ReasonCode::IoUnsupported};
          });
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-inline-outcomes", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(stopped.code() == rund::ReasonCode::NetPeerHandlerStopped);
  TEST_ASSERT(HasCounts(stopped, 0u, 0u, 1u));
  TEST_ASSERT(task_failed.code() == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(HasCounts(task_failed, 0u, 1u, 0u));
  TEST_ASSERT(invocation_failed.code() ==
              rund::ReasonCode::NetPeerHandlerFailed);
  TEST_ASSERT(HasCounts(invocation_failed, 0u, 1u, 0u));
  TEST_ASSERT(flattened_failed.code() == rund::ReasonCode::IoUnsupported);
  TEST_ASSERT(HasCounts(flattened_failed, 0u, 1u, 0u));
  TEST_ASSERT(stopped_close.ok());
  TEST_ASSERT(task_failed_close.ok());
  TEST_ASSERT(invocation_failed_close.ok());
  TEST_ASSERT(flattened_failed_close.ok());
  return 0;
}

[[nodiscard]] int RunServerTaskFailureCase() {
  rund::net::server::Result result{};
  rund::task::Status joined{};
  const rund::Session::Result run = rund::run(NetServerRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      result = co_await rund::net::server::Task{
          rund::task::Task<rund::net::server::Result>{
              rund::ReasonCode::TaskCapacityExceeded}};
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-task-failure", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!result);
  TEST_ASSERT(result.code() == rund::ReasonCode::TaskCapacityExceeded);
  return 0;
}

[[nodiscard]] int RunServerInlineWouldBlockCase() {
  LoopbackFixture fixture{};
  TEST_ASSERT(PrepareLoopbackListener(fixture) == 0);
  ServerSocketCleanup client{};
  TEST_ASSERT(StartLoopbackClient(fixture.address, client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 2u;
  rund::net::server::Result served{};
  rund::task::Status joined{};
  const rund::Session::Result run = rund::run(NetServerRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      served = co_await rund::net::server::serve(
          options,
          [](rund::net::server::Peer peer)
              -> rund::task::Task<rund::net::server::PeerResult> {
            const rund::net::CloseResult closed = peer.socket.close();
            co_return closed
                ? rund::net::server::PeerResult::complete()
                : rund::net::server::PeerResult::fail(closed.code());
          });
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-inline-would-block", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 1u);
  TEST_ASSERT(served.completed == 1u);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.would_block);
  TEST_ASSERT(!served.budget_exhausted);
  TEST_ASSERT(CountEvents(run, rund::host::EventKind::NetAccept) == 2u);
  return 0;
}

[[nodiscard]] int RunServerInlineArrivalCase() {
  LoopbackFixture fixture{};
  TEST_ASSERT(PrepareLoopbackListener(fixture) == 0);
  ServerSocketCleanup first_client{};
  ServerSocketCleanup second_client{};
  TEST_ASSERT(StartLoopbackClient(fixture.address, first_client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 2u;
  bool first_suspended = false;
  bool second_connected = false;
  bool arrived_while_suspended = false;
  std::uint32_t handled = 0u;
  rund::net::server::Result served{};
  rund::task::Status server_joined{};
  rund::task::Status connector_joined{};
  const rund::Session::Result run = rund::run(NetServerRunSpec(), [&] {
    auto connector = [&]() -> rund::task::Task<void> {
      while (!first_suspended) {
        (void)co_await rund::task::yield();
      }
      TEST_ASSERT(StartLoopbackClient(fixture.address, second_client) == 0);
      arrived_while_suspended = first_suspended;
      second_connected = true;
    };
    const rund::task::Handle connector_task =
        rund::task::spawn("net-server-inline-connect", connector());

    auto scenario = [&]() -> rund::task::Task<void> {
      served = co_await rund::net::server::serve(
          options,
          [&](rund::net::server::Peer peer)
              -> rund::task::Task<rund::net::server::PeerResult> {
            if (handled == 0u) {
              first_suspended = true;
              while (!second_connected) {
                (void)co_await rund::task::yield();
              }
              first_suspended = false;
            }
            ++handled;
            const rund::net::CloseResult closed = peer.socket.close();
            co_return closed
                ? rund::net::server::PeerResult::complete()
                : rund::net::server::PeerResult::fail(closed.code());
          });
    };
    const rund::task::Handle server_task =
        rund::task::spawn("net-server-inline-arrival", scenario());
    server_joined = rund::task::join(server_task);
    connector_joined = rund::task::join(connector_task);
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(server_joined.ok());
  TEST_ASSERT(connector_joined.ok());
  TEST_ASSERT(arrived_while_suspended);
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == 2u);
  TEST_ASSERT(served.started == 2u);
  TEST_ASSERT(served.completed == 2u);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.budget_exhausted);
  TEST_ASSERT(handled == 2u);
  TEST_ASSERT(CountEvents(run, rund::host::EventKind::NetAccept) == 2u);
  return 0;
}

[[nodiscard]] int RunServerPublicIoCase() {
  auto opened_listener = rund::net::open();
  TEST_ASSERT(opened_listener.ok());
  rund::net::Socket listener = std::move(opened_listener.socket);
  TEST_ASSERT(rund::net::bind(
                  listener.view(),
                  rund::net::Address::loopback(rund::net::Family::IPv4))
                  .ok());
  TEST_ASSERT(rund::net::listen(listener.view(), 1).ok());
  const auto local = rund::net::local(listener.view());
  TEST_ASSERT(local.ok());

  auto opened_client = rund::net::open();
  TEST_ASSERT(opened_client.ok());
  rund::net::Socket client = std::move(opened_client.socket);
  const std::array payload{std::byte{0x2au}};
  PublicClient client_result{};
  std::byte peer_byte{};
  rund::net::server::Result served{};
  rund::task::Status client_joined{};
  rund::task::Status server_joined{};

  const rund::Session::Result run = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        auto server = [&]() -> rund::task::Task<void> {
          served = co_await rund::net::server::serve(
              rund::net::server::Options{
                  .listener = listener.view(),
                  .accepts = {.max_accepts = 1u},
                  .task_name = "server.peer",
              },
              [&](rund::net::server::Peer peer) {
                return ReceivePublicByte(std::move(peer), peer_byte);
              });
        };
        const auto server_task = rund::task::spawn("server", server());
        const auto client_task = rund::task::spawn(
            "client", SendPublicByte(client.view(), local.address, payload,
                                      client_result));
        client_joined = rund::task::join(client_task);
        server_joined = rund::task::join(server_task);
      });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(client_joined.ok());
  TEST_ASSERT(server_joined.ok());
  TEST_ASSERT(served.ok());
  TEST_ASSERT(client_result.started.ok());
  TEST_ASSERT(client_result.connected.ok());
  TEST_ASSERT(client_result.sent.ok());
  TEST_ASSERT(client_result.sent.bytes == 1);
  TEST_ASSERT(peer_byte == payload[0]);
  TEST_ASSERT(HasCounts(served, 1u, 0u, 0u));
  TEST_ASSERT(served.budget_exhausted);
  return 0;
}

} // namespace

int RunServerInlineLoopbackCase() {
  constexpr std::size_t kClientsPerBatch = 2u;
  constexpr std::size_t kClients = kClientsPerBatch * 2u;
  LoopbackFixture fixture{};
  TEST_ASSERT(PrepareLoopbackListener(fixture) == 0);
  std::array<ServerSocketCleanup, kClients> client_cleanup{};
  std::array<std::byte, kClients> expected_bytes{};
  for (std::size_t index = 0; index < kClients; ++index) {
    expected_bytes[index] = static_cast<std::byte>(0x41u + index);
    TEST_ASSERT(StartLoopbackClientWithByte(fixture.address,
                                            expected_bytes[index],
                                            client_cleanup[index]) == 0);
  }

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = static_cast<std::uint32_t>(kClientsPerBatch);

  std::array<std::byte, kClients> observed_bytes{};
  std::array<rund::net::ready::Ticket, kClients> ready_results{};
  std::array<rund::net::ReceiveResult, kClients> recv_results{};
  std::array<rund::net::CloseResult, kClients> close_results{};
  std::uint64_t callback_count = 0u;
  bool accepted_native_valid = true;
  rund::net::server::Result warmed{};
  rund::net::server::Result served{};
  rund::task::Status joined{};
  std::uint64_t warm_allocations = ~std::uint64_t{0u};
  const rund::Session::Result report = rund::run(NetServerRunSpec(), [&] {
    auto serve = [&]() -> rund::task::Task<void> {
      const auto make_handler = [&] {
        return [marker = std::make_unique<bool>(true), &callback_count,
                &accepted_native_valid, &ready_results, &recv_results,
                &observed_bytes,
                &close_results](rund::net::server::Peer peer) mutable
                   -> rund::task::Task<rund::net::server::PeerResult> {
          if (marker == nullptr || callback_count >= kClients) {
            co_return rund::net::server::PeerResult::stop();
          }
          const std::size_t index = static_cast<std::size_t>(callback_count);
          ++callback_count;
          accepted_native_valid =
              accepted_native_valid &&
              rund::node::test::net::native(peer.socket) >= 0 &&
              rund::node::test::net::generation(peer.socket) != 0u;
          ready_results[index] = co_await rund::net::ready::timed::read(
              peer.socket.view(), std::chrono::milliseconds{100});
          if (ready_results[index].ok() && ready_results[index].ready()) {
            std::array<std::byte, 1u> buffer{};
            recv_results[index] = rund::net::receive(
                std::move(ready_results[index]), std::span<std::byte>{buffer});
            if (recv_results[index].ok() && recv_results[index].bytes == 1) {
              observed_bytes[index] = buffer[0];
            }
          }
          close_results[index] = peer.socket.close();
          if (!ready_results[index]) {
            co_return rund::net::server::PeerResult::fail(
                ready_results[index].code());
          }
          if (!recv_results[index]) {
            co_return rund::net::server::PeerResult::fail(
                recv_results[index].code());
          }
          co_return close_results[index]
              ? rund::net::server::PeerResult::complete()
              : rund::net::server::PeerResult::fail(
                    close_results[index].code());
        };
      };

      auto first_handler = make_handler();
      warmed =
          co_await rund::net::server::serve(options, std::move(first_handler));

      auto second_handler = make_handler();
      runtime_task_allocation::Start();
      served =
          co_await rund::net::server::serve(options, std::move(second_handler));
      runtime_task_allocation::Stop();
      warm_allocations = runtime_task_allocation::Count();
    };
    const rund::task::Handle server =
        rund::task::spawn("net-server-inline", serve());
    joined = rund::task::join(server);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(warmed.ok());
  TEST_ASSERT(warmed.accepted == kClientsPerBatch);
  TEST_ASSERT(warmed.started == kClientsPerBatch);
  TEST_ASSERT(warmed.completed == kClientsPerBatch);
  TEST_ASSERT(warmed.failed == 0u);
  TEST_ASSERT(warmed.stopped == 0u);
  TEST_ASSERT(warmed.rejected == 0u);
  TEST_ASSERT(warmed.budget_exhausted);
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == kClientsPerBatch);
  TEST_ASSERT(served.started == kClientsPerBatch);
  TEST_ASSERT(served.completed == kClientsPerBatch);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.rejected == 0u);
  TEST_ASSERT(served.budget_exhausted);
  TEST_ASSERT(warm_allocations == 0u);
  TEST_ASSERT(report.tasks().coroutine_tasks_admitted() == 7u);
  // Two listener registrations (one per batch) plus one peer read per client.
  TEST_ASSERT(CountEvents(report, rund::host::EventKind::IoReady) ==
              kClients + 2u);
  TEST_ASSERT(CountEvents(report, rund::host::EventKind::NetAccept) ==
              kClients);
  TEST_ASSERT(callback_count == kClients);
  TEST_ASSERT(accepted_native_valid);
  for (std::size_t index = 0; index < kClients; ++index) {
    TEST_ASSERT(ready_results[index].ok());
    TEST_ASSERT(ready_results[index].consumed());
    TEST_ASSERT(!ready_results[index].timed_out());
    TEST_ASSERT(recv_results[index].ok());
    TEST_ASSERT(recv_results[index].bytes == 1);
    TEST_ASSERT(close_results[index].ok());
  }
  std::array<bool, kClients> seen_expected_bytes{};
  for (const std::byte observed : observed_bytes) {
    bool matched = false;
    for (std::size_t index = 0; index < kClients; ++index) {
      if (!seen_expected_bytes[index] && observed == expected_bytes[index]) {
        seen_expected_bytes[index] = true;
        matched = true;
        break;
      }
    }
    TEST_ASSERT(matched);
  }
  TEST_ASSERT(RunServerInlineOutcomeCase() == 0);
  TEST_ASSERT(RunServerTaskFailureCase() == 0);
  TEST_ASSERT(RunServerInlineWouldBlockCase() == 0);
  TEST_ASSERT(RunServerInlineArrivalCase() == 0);
  TEST_ASSERT(RunServerPublicIoCase() == 0);
  return 0;
}
