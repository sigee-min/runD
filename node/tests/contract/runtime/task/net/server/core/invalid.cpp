#include "local.hpp"
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/result.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/task/api.hpp>

#include "../../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

static_assert(noexcept(rund::net::server::PeerResult::complete()));
static_assert(noexcept(rund::net::server::PeerResult::stop()));
static_assert(noexcept(
    rund::net::server::PeerResult::fail(rund::ReasonCode::IoUnsupported)));
static_assert(std::is_trivially_copyable_v<rund::net::server::PeerResult>);
static_assert(sizeof(rund::net::server::PeerResult) ==
              sizeof(rund::net::Status));
static_assert(rund::net::server::detail::classify_peer_terminal(
                  rund::net::server::PeerResult::complete()) ==
              rund::net::server::detail::PeerTerminalClass::Completed);
static_assert(rund::net::server::detail::classify_peer_terminal(
                  rund::net::server::PeerResult::stop()) ==
              rund::net::server::detail::PeerTerminalClass::Stopped);
static_assert(rund::net::server::detail::classify_peer_terminal(
                  rund::net::server::PeerResult{}) ==
              rund::net::server::detail::PeerTerminalClass::Failed);

int RunServerInvalidListenerCase() {
  runtime_task_allocation::Start();
  const rund::net::server::PeerResult complete =
      rund::net::server::PeerResult::complete();
  const rund::net::server::PeerResult stopped =
      rund::net::server::PeerResult::stop();
  const rund::net::server::PeerResult failed =
      rund::net::server::PeerResult::fail(rund::ReasonCode::IoUnsupported);
  runtime_task_allocation::Stop();
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(complete.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(stopped.code() == rund::ReasonCode::NetPeerHandlerStopped);
  TEST_ASSERT(failed.code() == rund::ReasonCode::IoUnsupported);
  TEST_ASSERT(rund::net::server::detail::classify_peer_terminal(failed) ==
              rund::net::server::detail::PeerTerminalClass::Failed);
  TEST_ASSERT(
      rund::net::server::PeerResult::fail(rund::ReasonCode::Ok).code() ==
      rund::ReasonCode::NetPeerHandlerFailed);
  TEST_ASSERT(rund::net::server::detail::classify_peer_terminal(
                  rund::net::server::PeerResult::fail(rund::ReasonCode::Ok)) ==
              rund::net::server::detail::PeerTerminalClass::Failed);
  TEST_ASSERT(rund::net::server::PeerResult::fail(
                  rund::ReasonCode::NetPeerHandlerStopped)
                  .code() == rund::ReasonCode::NetPeerHandlerFailed);
  TEST_ASSERT(rund::net::server::detail::classify_peer_terminal(
                  rund::net::server::PeerResult::fail(
                      rund::ReasonCode::NetPeerHandlerStopped)) ==
              rund::net::server::detail::PeerTerminalClass::Failed);
  TEST_ASSERT(rund::net::server::PeerResult::fail(
                  static_cast<rund::ReasonCode>(
                      std::numeric_limits<std::uint16_t>::max()))
                  .code() == rund::ReasonCode::NetPeerHandlerFailed);
  TEST_ASSERT(
      rund::net::server::detail::classify_peer_terminal(
          rund::net::server::PeerResult::fail(static_cast<rund::ReasonCode>(
              std::numeric_limits<std::uint16_t>::max()))) ==
      rund::net::server::detail::PeerTerminalClass::Failed);

  bool callback_ran = false;
  rund::net::server::Options options{};
  options.accepts.max_accepts = 1u;
  rund::net::server::Result result{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(NetServerRunSpec(), [&] {
    auto scenario = [&]() -> rund::task::Task<void> {
      auto handler = [&](rund::net::server::Peer peer)
          -> rund::task::Task<rund::net::server::PeerResult> {
        (void)peer;
        callback_ran = true;
        co_return rund::net::server::PeerResult::complete();
      };
      result = co_await rund::net::server::serve(options, std::move(handler));
    };
    const rund::task::Handle task =
        rund::task::spawn("net-server-invalid", scenario());
    joined = rund::task::join(task);
  });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(result.accepted == 0u);
  TEST_ASSERT(result.started == 0u);
  TEST_ASSERT(result.completed == 0u);
  TEST_ASSERT(result.failed == 0u);
  TEST_ASSERT(result.stopped == 0u);
  TEST_ASSERT(result.rejected == 0u);
  TEST_ASSERT(!callback_ran);
  return 0;
}
