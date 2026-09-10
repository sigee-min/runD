#include "local.hpp"

#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/net/server/task.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

namespace server_inline_detail {
namespace {

struct SynchronousThrow final {
  rund::net::CloseResult *closed = nullptr;

  [[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
  operator()(rund::net::server::Peer peer) const {
    *closed = peer.socket.close();
    throw 1;
  }
};

} // namespace

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

} // namespace server_inline_detail
