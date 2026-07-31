#include "local.hpp"

#include <rund/net/accept.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/drain.hpp>
#include <rund/net/handoff.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

int RunNetMultiClientSubstrateCase() {
  constexpr std::size_t kClients = 64u;
  MultiClientLoopbackFixture fixture{};
  TEST_ASSERT(PrepareMultiClientLoopbackListener(fixture) == 0);

  std::array<std::byte, kClients> payload{};
  std::array<MultiClientSocketCleanup, kClients> clients{};
  for (std::size_t index = 0; index < kClients; ++index) {
    payload[index] = static_cast<std::byte>(0x30u + (index & 0x3fu));
    TEST_ASSERT(StartMultiClientBlockingClient(fixture.connect_address,
                                               payload[index],
                                               clients[index]) == 0);
  }

  std::vector<rund::net::Socket> accepted_sockets{};
  accepted_sockets.reserve(kClients);
  std::vector<rund::task::Handle> connection_tasks{};
  connection_tasks.reserve(kClients);
  std::array<rund::net::accept::Prepared, kClients> handoffs{};
  std::array<rund::net::ready::Ticket, kClients> ready_results{};
  std::array<rund::net::ReceiveResult, kClients> recv_results{};
  std::array<rund::net::drain::WriteResult, kClients> write_results{};
  std::array<rund::net::CloseResult, kClients> close_results{};
  std::array<std::byte, kClients> echoed_by_server{};
  rund::net::accept::Drain accepted{};
  rund::task::Status connection_joined{};
  rund::task::Status server_joined{};

  auto connection = [&](const std::size_t index) -> rund::task::Task<void> {
    rund::net::Socket socket = std::move(accepted_sockets[index]);
    ready_results[index] = co_await rund::net::ready::timed::read(
        socket.view(), std::chrono::milliseconds{100});
    if (ready_results[index].ok() && ready_results[index].ready()) {
      std::array<std::byte, 1u> byte{};
      recv_results[index] = rund::net::receive(std::move(ready_results[index]),
                                               std::span<std::byte>{byte});
      if (recv_results[index].ok()) {
        echoed_by_server[index] = byte[0];
        auto writable = co_await rund::net::ready::write(socket.view());
        write_results[index] = rund::net::drain::write(
            std::move(writable), std::span<const std::byte>{byte},
            rund::net::drain::Budget{.max_operations = 1u});
      }
    }
    close_results[index] = socket.close();
  };
  auto serve = [&]() -> rund::task::Task<void> {
    const auto listener_ready =
        co_await rund::net::ready::read(fixture.listener.view());
    if (!listener_ready.ok()) {
      co_return;
    }
    const auto accept_result = co_await rund::net::accept::drain(
        fixture.listener.view(),
        rund::net::accept::Budget{.max_accepts =
                                      static_cast<std::uint32_t>(kClients)},
        [&](rund::net::accept::Result &&accept_result) {
          if (!accept_result.ok() || accepted_sockets.size() >= kClients) {
            return false;
          }
          const std::size_t index = accepted_sockets.size();
          handoffs[index] =
              rund::net::accept::prepare(std::move(accept_result));
          if (!handoffs[index].ok()) {
            return false;
          }
          accepted_sockets.push_back(std::move(handoffs[index].socket));
          connection_tasks.push_back(rund::task::spawn(
              "net-multi-client-connection", connection(index)));
          return true;
        });
    if (accept_result) {
      accepted = *accept_result;
    }
    connection_joined = rund::task::Status::success();
    for (const rund::task::Handle &handle : connection_tasks) {
      const rund::task::Status joined = co_await handle;
      if (!joined.ok() && connection_joined.ok()) {
        connection_joined = joined;
      }
    }
  };

  const rund::Session::Result report = rund::run(NetMultiClientRunSpec(), [&] {
    const rund::task::Handle server =
        rund::task::spawn("net-multi-client-server", serve());
    server_joined = rund::task::join(server);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(server_joined.ok());
  TEST_ASSERT(connection_joined.ok());
  TEST_ASSERT(accepted.ok());
  TEST_ASSERT(accepted.accepts == kClients);
  TEST_ASSERT(accepted_sockets.size() == kClients);
  for (std::size_t index = 0; index < kClients; ++index) {
    TEST_ASSERT(handoffs[index].ok());
    TEST_ASSERT(ready_results[index].ok());
    TEST_ASSERT(ready_results[index].consumed());
    TEST_ASSERT(!ready_results[index].ready());
    TEST_ASSERT(!ready_results[index].timed_out());
    TEST_ASSERT(recv_results[index].ok());
    TEST_ASSERT(recv_results[index].bytes == 1);
    TEST_ASSERT(echoed_by_server[index] == payload[index]);
    TEST_ASSERT(write_results[index].ok());
    TEST_ASSERT(write_results[index].all_written);
    TEST_ASSERT(write_results[index].bytes == 1u);
    TEST_ASSERT(close_results[index].ok());

    std::byte echoed{};
    TEST_ASSERT(RecvMultiClientOneWithTimeout(clients[index].fd, echoed));
    TEST_ASSERT(echoed == payload[index]);
  }
  return 0;
}
