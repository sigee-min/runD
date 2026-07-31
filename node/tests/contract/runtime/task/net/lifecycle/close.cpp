#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/io.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <span>

int RunNetLifecycleCloseInvalidatesWaitCase() {
  using namespace rund::node::test_contract::net_lifecycle;

  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket left = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket right = rund::node::test::net::admit(cleanup.right);
  TEST_ASSERT(rund::net::nonblocking(left.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(right.view(), true).ok());

  std::array<std::byte, 1u> bytes{};
  rund::net::ReceiveResult readable_result{};
  rund::net::CloseResult close_result{};
  rund::net::CloseResult stale_close_result{};
  rund::task::Status yield_result{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(RunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      readable_result =
          co_await rund::net::receive(left.view(), std::span<std::byte>{bytes});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-lifecycle-readable", wait());
    auto close = [&]() -> rund::task::Task<void> {
      yield_result = co_await rund::task::yield();
      close_result = left.close();
    };
    const rund::task::Handle closer =
        rund::task::spawn("net-lifecycle-close", close());
    joined = rund::task::join(waiter, closer);
  });

  stale_close_result = left.close();

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(yield_result.ok());
  TEST_ASSERT(close_result.ok());
  TEST_ASSERT(!readable_result.ok());
  TEST_ASSERT(readable_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(report.tasks().network().recv_calls() == 0u);
  TEST_ASSERT(report.tasks().reactor().close_invalidated_waits() == 1u);
  for (const rund::host::Event &event : report.events()) {
    TEST_ASSERT(event.kind != rund::host::EventKind::NetRecv);
  }
  TEST_ASSERT(!stale_close_result.ok());
  TEST_ASSERT(stale_close_result.code() == rund::ReasonCode::IoFdInvalid);
  return 0;
}
