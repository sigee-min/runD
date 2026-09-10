#include "local.hpp"

#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

namespace server_inline_detail {

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

} // namespace server_inline_detail
