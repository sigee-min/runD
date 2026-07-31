#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/result.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/channel.hpp>

#include "../../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

namespace {

[[nodiscard]] std::uint64_t
CountReadinessEvents(const rund::Session::Result &run) noexcept {
  std::uint64_t count = 0u;
  for (const rund::host::Event &event : run.events()) {
    count += event.kind == rund::host::EventKind::IoReady ? 1u : 0u;
  }
  return count;
}

[[nodiscard]] rund::SessionConfig FrameCapacityRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 4u,
      .scheduler =
          {
              .task_capacity = 6u,
              .ready_queue_capacity = 8u,
              .channel_capacity = 1u,
              .channel_buffer_capacity = 0u,
              .channel_wait_capacity = 2u,
              .reactor_wait_capacity = 4u,
              .observation_capacity = 32u,
              .host_event_capacity = 32u,
          },
  };
}

} // namespace

int RunServerParallelSuccessCase() {
  constexpr std::size_t kClientsPerBatch = 3u;
  constexpr std::size_t kClients = kServerParallelWarmClientEnvelope;
  static_assert(kClients == kClientsPerBatch * 2u);
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);

  std::array<ServerParallelSocketCleanup, kClients> client_cleanup{};
  for (std::size_t index = 0; index < kClients; ++index) {
    TEST_ASSERT(StartServerParallelClientWithByte(
                    fixture.address, static_cast<std::byte>(0x41u + index),
                    client_cleanup[index]) == 0);
  }

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = static_cast<std::uint32_t>(kClientsPerBatch);

  std::array<rund::task::Handle, kClientsPerBatch> peer_tasks{};
  std::array<rund::net::ready::Ticket, kClients> ready_results{};
  std::array<rund::net::ReceiveResult, kClients> recv_results{};
  std::array<rund::net::CloseResult, kClients> close_results{};
  std::atomic<std::uint64_t> callback_count{0u};
  rund::net::server::Result warmed{};
  rund::net::server::Result served{};
  rund::task::Status joined{};
  std::uint64_t warm_allocations = ~std::uint64_t{0u};

  const rund::Session::Result run = rund::run(NetServerParallelRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      const auto make_handler = [&] {
        return [marker = std::make_unique<bool>(true), &callback_count,
                &ready_results, &recv_results,
                &close_results](rund::net::server::Peer peer)
                   -> rund::task::Task<rund::net::server::PeerResult> {
          const std::size_t index = static_cast<std::size_t>(
              callback_count.fetch_add(1u, std::memory_order_relaxed));
          if (!*marker || index >= kClients) {
            co_return rund::net::server::PeerResult::stop();
          }
          ready_results[index] = co_await rund::net::ready::timed::read(
              peer.socket.view(), std::chrono::milliseconds{100});
          if (ready_results[index].ok() && ready_results[index].ready()) {
            std::array<std::byte, 1u> buffer{};
            recv_results[index] = rund::net::receive(
                rund::node::test::net::ticket(
                    peer.socket.view(), rund::net::ready::Interest::Readable),
                std::span<std::byte>{buffer});
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
      using Handler = decltype(make_handler());
      static_assert(!std::is_copy_constructible_v<Handler>);

      Handler first_handler = make_handler();
      warmed = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(first_handler));

      Handler second_handler = make_handler();
      runtime_task_allocation::Start();
      served = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(second_handler));
      runtime_task_allocation::Stop();
      warm_allocations = runtime_task_allocation::Count();
    };
    const rund::task::Handle server =
        rund::task::spawn("net-server-parallel", scenario());
    joined = rund::task::join(server);
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(warmed.ok());
  TEST_ASSERT(warmed.accepted == kClientsPerBatch);
  TEST_ASSERT(warmed.started == kClientsPerBatch);
  TEST_ASSERT(warmed.completed == kClientsPerBatch);
  TEST_ASSERT(warmed.failed == 0u);
  TEST_ASSERT(warmed.stopped == 0u);
  TEST_ASSERT(warmed.rejected == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(warmed, options.accepts.max_accepts));
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == kClientsPerBatch);
  TEST_ASSERT(served.started == kClientsPerBatch);
  TEST_ASSERT(served.completed == kClientsPerBatch);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.rejected == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(served, options.accepts.max_accepts));
  TEST_ASSERT(warm_allocations == 0u);
  TEST_ASSERT(run.tasks().coroutine_tasks_admitted() == 17u);
  // Two listener registrations (one per batch) plus one peer read per client.
  TEST_ASSERT(CountReadinessEvents(run) == kClients + 2u);
  TEST_ASSERT(callback_count.load(std::memory_order_relaxed) == kClients);
  for (std::size_t index = 0; index < kClients; ++index) {
    TEST_ASSERT(ready_results[index].ok());
    TEST_ASSERT(ready_results[index].ready());
    TEST_ASSERT(!ready_results[index].timed_out());
    TEST_ASSERT(recv_results[index].ok());
    TEST_ASSERT(recv_results[index].bytes == 1u);
    TEST_ASSERT(close_results[index].ok());
  }
  return 0;
}

int RunServerParallelFrameCapacityCase() {
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);
  ServerParallelSocketCleanup client{};
  TEST_ASSERT(StartServerParallelSingleClient(fixture.address, client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 1u;
  std::array<rund::task::Handle, 1u> peer_tasks{};
  rund::net::server::Result served{};
  rund::task::Status server_joined{};
  rund::task::Status release_joined{};
  rund::task::Status sent{};

  const rund::Session::Result run = rund::run(FrameCapacityRunSpec(), [&] {
    auto gate = rund::task::channel<int>::make(0u);
    auto release = [&]() -> rund::task::Task<void> {
      sent = co_await gate.send(1);
    };
    const rund::task::Handle release_task =
        rund::task::spawn("net-server-parallel-release", release());

    auto scenario = [&]() -> rund::task::Task<void> {
      auto handler = [&](rund::net::server::Peer peer)
          -> rund::task::Task<rund::net::server::PeerResult> {
        const rund::task::ReceiveResult<int> released = co_await gate.recv();
        const rund::net::CloseResult closed = peer.socket.close();
        if (!released) {
          co_return rund::net::server::PeerResult::fail(released.code());
        }
        if (*released != 1) {
          co_return rund::net::server::PeerResult::fail(
              rund::ReasonCode::NetPeerHandlerFailed);
        }
        co_return closed ? rund::net::server::PeerResult::complete()
                         : rund::net::server::PeerResult::fail(closed.code());
      };
      served = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(handler));
    };
    const rund::task::Handle server =
        rund::task::spawn("net-server-parallel-frame-capacity", scenario());
    server_joined = rund::task::join(server);
    release_joined = rund::task::join(release_task);
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(server_joined.ok());
  TEST_ASSERT(release_joined.ok());
  TEST_ASSERT(sent.ok());
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 1u);
  TEST_ASSERT(served.completed == 1u);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.rejected == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(served, options.accepts.max_accepts));
  TEST_ASSERT(run.tasks().resources().coroutine_frame_failures() == 0u);
  TEST_ASSERT(run.tasks().coroutine_tasks_admitted() == 6u);
  TEST_ASSERT(CountReadinessEvents(run) == 1u);
  TEST_ASSERT(run.tasks().resources().coroutine_frames_high_water() == 5u);
  return 0;
}
