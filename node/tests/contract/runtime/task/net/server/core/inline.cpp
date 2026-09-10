#include "inline/local.hpp"
#include "local.hpp"
#include "src/host/net/test/socket.hpp"
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
  TEST_ASSERT(server_inline_detail::CountEvents(
                  report, rund::host::EventKind::IoReady) == kClients + 2u);
  TEST_ASSERT(server_inline_detail::CountEvents(
                  report, rund::host::EventKind::NetAccept) == kClients);
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
  TEST_ASSERT(server_inline_detail::RunServerInlineOutcomeCase() == 0);
  TEST_ASSERT(server_inline_detail::RunServerTaskFailureCase() == 0);
  TEST_ASSERT(server_inline_detail::RunServerInlineWouldBlockCase() == 0);
  TEST_ASSERT(server_inline_detail::RunServerInlineArrivalCase() == 0);
  TEST_ASSERT(server_inline_detail::RunServerPublicIoCase() == 0);
  return 0;
}
