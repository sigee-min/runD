#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/result.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <atomic>
#include <span>
#include <string_view>
#include <utility>

namespace {

template <typename Handler>
[[nodiscard]] int
RunServerParallelPeerFailureCase(const char *const task_name, Handler handler,
                                 const rund::ReasonCode expected) {
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);
  ServerParallelSocketCleanup client{};
  TEST_ASSERT(StartServerParallelSingleClient(fixture.address, client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 1u;
  std::array<rund::task::Handle, 1u> peer_tasks{};

  rund::net::server::Result served{};
  rund::task::Status joined{};
  const rund::Session::Result run = rund::run(NetServerParallelRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      served = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(handler));
    };
    const rund::task::Handle server = rund::task::spawn(task_name, scenario());
    joined = rund::task::join(server);
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!served.ok());
  const bool stopped = expected == rund::ReasonCode::NetPeerHandlerStopped;
  TEST_ASSERT(std::string_view{served.error()} == rund::ReasonString(expected));
  TEST_ASSERT(served.code() == expected);
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 1u);
  TEST_ASSERT(served.completed == 0u);
  TEST_ASSERT(served.failed == (stopped ? 0u : 1u));
  TEST_ASSERT(served.stopped == (stopped ? 1u : 0u));
  TEST_ASSERT(served.rejected == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(served, options.accepts.max_accepts));
  return 0;
}

struct SynchronousThrowHandler final {
  rund::net::CloseResult *closed = nullptr;

  [[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
  operator()(rund::net::server::Peer peer) const {
    *closed = peer.socket.close();
    throw 1;
  }
};

struct ClosePeerHandler final {
  [[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
  operator()(rund::net::server::Peer peer) const {
    const rund::net::CloseResult closed = peer.socket.close();
    co_return closed ? rund::net::server::PeerResult::complete()
                     : rund::net::server::PeerResult::fail(closed.code());
  }
};

[[nodiscard]] int RunServerParallelMissingTerminalCase() {
  rund::net::server::Result served{rund::ReasonCode::Ok};
  served.accepted = 1u;
  served.started = 1u;
  const rund::net::server::detail::Outcomes outcomes{.result = &served};

  rund::net::server::detail::finish(served, rund::task::Status::success(),
                                    outcomes);

  TEST_ASSERT(served.code() == rund::ReasonCode::NetPeerHandlerFailed);
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 1u);
  TEST_ASSERT(served.completed == 0u);
  TEST_ASSERT(served.failed == 1u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(served, 1u));
  return 0;
}

[[nodiscard]] bool MixedTerminalOrderSelectsLowest(
    const bool lower_index_first) noexcept {
  rund::net::server::Result served{rund::ReasonCode::Ok};
  served.accepted = 2u;
  served.started = 2u;
  rund::net::server::detail::Outcomes outcomes{.result = &served};
  const auto record_lower = [&] {
    rund::net::server::detail::record(
        outcomes, 0u, rund::net::server::PeerResult::stop());
  };
  const auto record_higher = [&] {
    rund::net::server::detail::record(
        outcomes, 1u, rund::net::server::PeerResult::fail(
                          rund::ReasonCode::IoUnsupported));
  };
  if (lower_index_first) {
    record_lower();
    record_higher();
  } else {
    record_higher();
    record_lower();
  }
  rund::net::server::detail::finish(
      served, rund::task::Status::success(), outcomes);
  return served.code() == rund::ReasonCode::NetPeerHandlerStopped &&
         served.completed == 0u && served.failed == 1u &&
         served.stopped == 1u && ServerCountsAreConsistent(served, 2u);
}

[[nodiscard]] int RunServerParallelFailureOrderCase() {
  constexpr std::size_t kClients = 2u;
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);
  std::array<ServerParallelSocketCleanup, kClients> clients{};
  TEST_ASSERT(StartServerParallelClientWithByte(
                  fixture.address, std::byte{0x41u}, clients[0]) == 0);
  TEST_ASSERT(StartServerParallelClientWithByte(
                  fixture.address, std::byte{0x42u}, clients[1]) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = kClients;
  std::array<rund::task::Handle, kClients> peer_tasks{};
  std::atomic<bool> higher_finished{false};
  rund::net::server::Result served{};
  rund::task::Status joined{};

  const rund::Session::Result run = rund::run(NetServerParallelRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      auto handler = [&](rund::net::server::Peer peer)
          -> rund::task::Task<rund::net::server::PeerResult> {
        const rund::net::ready::Ticket readable =
            co_await rund::net::ready::read(peer.socket.view());
        std::array<std::byte, 1u> bytes{};
        const rund::net::ReceiveResult received =
            readable
                ? rund::net::receive(rund::node::test::net::ticket(
                                         peer.socket.view(),
                                         rund::net::ready::Interest::Readable),
                                     bytes)
                : rund::net::ReceiveResult{};
        const rund::net::CloseResult closed = peer.socket.close();
        if (!readable) {
          co_return rund::net::server::PeerResult::fail(readable.code());
        }
        if (!received) {
          co_return rund::net::server::PeerResult::fail(received.code());
        }
        if (!closed) {
          co_return rund::net::server::PeerResult::fail(closed.code());
        }
        if (bytes[0] == std::byte{0x42u}) {
          higher_finished.store(true, std::memory_order_release);
          co_return rund::net::server::PeerResult::fail(
              rund::ReasonCode::IoUnsupported);
        }
        while (!higher_finished.load(std::memory_order_acquire)) {
          static_cast<void>(co_await rund::task::yield());
        }
        co_return rund::net::server::PeerResult::stop();
      };
      served = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(handler));
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-parallel-failure-order", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(served.code() == rund::ReasonCode::NetPeerHandlerStopped);
  TEST_ASSERT(served.accepted == kClients);
  TEST_ASSERT(served.started == kClients);
  TEST_ASSERT(served.completed == 0u);
  TEST_ASSERT(served.failed == 1u);
  TEST_ASSERT(served.stopped == 1u);
  TEST_ASSERT(ServerCountsAreConsistent(served, kClients));
  return 0;
}

[[nodiscard]] int RunServerParallelReadyQueueSaturationCase() {
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);
  ServerParallelSocketCleanup client{};
  TEST_ASSERT(StartServerParallelSingleClient(fixture.address, client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 1u;
  std::array<rund::task::Handle, 1u> peer_tasks{};
  rund::net::server::Result served{};
  rund::task::Status joined{};
  bool admission_rejected = false;

  const rund::Session::Result run = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 6u,
                  .ready_queue_capacity = 1u,
                  .reactor_wait_capacity = 2u,
                  .observation_capacity = 16u,
                  .host_event_capacity = 16u,
              },
      },
      [&] {
        auto scenario = [&]() -> rund::task::Task<void> {
          served = co_await rund::net::server::serve(options, peer_tasks,
                                                     ClosePeerHandler{});
        };
        const rund::task::Handle server =
            rund::task::spawn("net-server-parallel-ready", scenario());
        const rund::task::Handle overflow =
            rund::task::spawn("ready-admission-overflow", [] {});
        admission_rejected =
            !overflow &&
            overflow.code() == rund::ReasonCode::ReadyQueueCapacityExceeded;
        joined = rund::task::join(server);
      });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(admission_rejected);
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(served.ok());
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 1u);
  TEST_ASSERT(served.completed == 1u);
  TEST_ASSERT(served.failed == 0u);
  TEST_ASSERT(served.stopped == 0u);
  TEST_ASSERT(served.rejected == 0u);
  TEST_ASSERT(run.tasks().max_ready_depth() > 1u);
  TEST_ASSERT(run.tasks().max_ready_depth() <= 6u);
  return 0;
}

[[nodiscard]] int RunServerParallelTaskCapacityCase() {
  bool callback_ran = false;
  rund::net::server::Options options{};
  options.accepts.max_accepts = 1u;

  rund::net::server::Result insufficient{};
  rund::net::server::Result empty{};
  rund::task::Status joined{};
  const rund::Session::Result run = rund::run(NetServerParallelRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      auto handler = [&](rund::net::server::Peer)
          -> rund::task::Task<rund::net::server::PeerResult> {
        callback_ran = true;
        co_return rund::net::server::PeerResult::complete();
      };
      std::span<rund::task::Handle> no_peer_tasks{};
      insufficient =
          co_await rund::net::server::serve(options, no_peer_tasks, handler);

      options.accepts.max_accepts = 0u;
      empty = co_await rund::net::server::serve(options, no_peer_tasks,
                                                std::move(handler));
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-parallel-capacity", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!insufficient.ok());
  TEST_ASSERT(insufficient.code() == rund::ReasonCode::TaskCapacityExceeded);
  TEST_ASSERT(insufficient.accepted == 0u);
  TEST_ASSERT(insufficient.started == 0u);
  TEST_ASSERT(ServerCountsAreConsistent(insufficient, 1u));
  TEST_ASSERT(empty.ok());
  TEST_ASSERT(empty.accepted == 0u);
  TEST_ASSERT(empty.started == 0u);
  TEST_ASSERT(empty.completed == 0u);
  TEST_ASSERT(empty.failed == 0u);
  TEST_ASSERT(empty.stopped == 0u);
  TEST_ASSERT(empty.rejected == 0u);
  TEST_ASSERT(empty.budget_exhausted);
  TEST_ASSERT(ServerCountsAreConsistent(empty, 0u));
  TEST_ASSERT(!callback_ran);
  return 0;
}

[[nodiscard]] int RunServerParallelSpawnFailureCase() {
  ServerParallelLoopbackFixture fixture{};
  TEST_ASSERT(PrepareServerParallelLoopbackListener(fixture) == 0);
  ServerParallelSocketCleanup client{};
  TEST_ASSERT(StartServerParallelSingleClient(fixture.address, client) == 0);

  rund::net::server::Options options{};
  options.listener = fixture.listener.view();
  options.accepts.max_accepts = 1u;
  options.task_name = "";
  std::array<rund::task::Handle, 1u> peer_tasks{};
  bool callback_ran = false;
  rund::net::server::Result served{};
  rund::task::Status joined{};
  const rund::Session::Result run = rund::run(NetServerParallelRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      auto handler = [&](rund::net::server::Peer)
          -> rund::task::Task<rund::net::server::PeerResult> {
        callback_ran = true;
        co_return rund::net::server::PeerResult::complete();
      };
      served = co_await rund::net::server::serve(options, peer_tasks,
                                                 std::move(handler));
    };
    joined = rund::task::join(
        rund::task::spawn("net-server-parallel-spawn", scenario()));
  });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!served.ok());
  TEST_ASSERT(served.code() == rund::ReasonCode::NetPeerSpawnFailed);
  TEST_ASSERT(served.accepted == 1u);
  TEST_ASSERT(served.started == 0u);
  TEST_ASSERT(served.rejected == 1u);
  TEST_ASSERT(served.budget_exhausted);
  TEST_ASSERT(ServerCountsAreConsistent(served, options.accepts.max_accepts));
  TEST_ASSERT(!callback_ran);
  return 0;
}

} // namespace

int RunServerParallelHandlerFailureCase() {
  rund::net::CloseResult stopped_close{};
  TEST_ASSERT(RunServerParallelPeerFailureCase(
                  "net-server-parallel-false",
                  [&](rund::net::server::Peer peer)
                      -> rund::task::Task<rund::net::server::PeerResult> {
                    stopped_close = peer.socket.close();
                    co_return rund::net::server::PeerResult::stop();
                  },
                  rund::ReasonCode::NetPeerHandlerStopped) == 0);
  TEST_ASSERT(stopped_close.ok());

  rund::net::CloseResult task_failed_close{};
  TEST_ASSERT(RunServerParallelPeerFailureCase(
                  "net-server-parallel-task-failure",
                  [&](rund::net::server::Peer peer)
                      -> rund::task::Task<rund::net::server::PeerResult> {
                    task_failed_close = peer.socket.close();
                    throw 1;
                    co_return rund::net::server::PeerResult::complete();
                  },
                  rund::ReasonCode::TaskFailed) == 0);
  TEST_ASSERT(task_failed_close.ok());

  rund::net::CloseResult invocation_failed_close{};
  TEST_ASSERT(RunServerParallelPeerFailureCase(
                  "net-server-parallel-invoke-failure",
                  SynchronousThrowHandler{.closed = &invocation_failed_close},
                  rund::ReasonCode::NetPeerHandlerFailed) == 0);
  TEST_ASSERT(invocation_failed_close.ok());

  rund::net::CloseResult operation_failed_close{};
  TEST_ASSERT(RunServerParallelPeerFailureCase(
                  "net-server-parallel-operation-failure",
                  [&](rund::net::server::Peer peer)
                      -> rund::task::Task<rund::net::server::PeerResult> {
                    operation_failed_close = peer.socket.close();
                    co_return rund::net::server::PeerResult::fail(
                        rund::ReasonCode::IoUnsupported);
                  },
                  rund::ReasonCode::IoUnsupported) == 0);
  TEST_ASSERT(operation_failed_close.ok());
  TEST_ASSERT(RunServerParallelMissingTerminalCase() == 0);
  TEST_ASSERT(MixedTerminalOrderSelectsLowest(false));
  TEST_ASSERT(MixedTerminalOrderSelectsLowest(true));
  TEST_ASSERT(RunServerParallelFailureOrderCase() == 0);
  TEST_ASSERT(RunServerParallelReadyQueueSaturationCase() == 0);
  TEST_ASSERT(RunServerParallelTaskCapacityCase() == 0);
  TEST_ASSERT(RunServerParallelSpawnFailureCase() == 0);
  return 0;
}
