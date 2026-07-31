#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/accept.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/handoff.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <span>
#include <utility>

int RunNetAcceptHandoffPreparedSocketCase() {
  using namespace rund::node::test_contract::net_accept_handoff;

  LoopbackFixture fixture{};
  TEST_ASSERT(PrepareLoopbackListener(fixture) == 0);
  rund::net::Socket client{};
  TEST_ASSERT(StartNonblockingClient(fixture.connect_address, client) == 0);

  std::array<std::byte, 1u> one_byte_buffer{};
  rund::net::accept::Drain drained{};
  rund::net::accept::Prepared handoff{};
  rund::net::ReceiveResult pre_payload_read{};
  rund::net::ready::Ticket listener_ready{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(RunSpec(), [&] {
    auto accept = [&]() -> rund::task::Task<void> {
      listener_ready = co_await rund::net::ready::read(fixture.listener.view());
      if (!listener_ready.ok()) {
        co_return;
      }
      const rund::net::accept::Drain result = rund::net::accept::drain(
          std::move(listener_ready),
          rund::net::accept::Budget{.max_accepts = 1u},
          [&](rund::net::accept::Result &&accepted) {
            if (!accepted.ok()) {
              return false;
            }
            handoff = rund::net::accept::prepare(std::move(accepted));
            if (!handoff.ok()) {
              return false;
            }
            pre_payload_read =
                rund::net::receive(rund::node::test::net::ticket(
                                       handoff.socket.view(),
                                       rund::net::ready::Interest::Readable),
                                   std::span<std::byte>{one_byte_buffer});
            return true;
          });
      drained = result;
    };
    const rund::task::Handle task =
        rund::task::spawn("net-accept-handoff", accept());
    joined = rund::task::join(task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(listener_ready.ok());
  TEST_ASSERT(listener_ready.consumed());
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.accepts == 1u);
  TEST_ASSERT(handoff.ok());
  TEST_ASSERT(rund::node::test::net::native(handoff.socket) >= 0);
  TEST_ASSERT(rund::node::test::net::generation(handoff.socket) != 0u);
  TEST_ASSERT(handoff.nonblocking.ok());
  TEST_ASSERT(!pre_payload_read.ok());
  TEST_ASSERT(pre_payload_read.code() == rund::ReasonCode::IoWouldBlock);
  return 0;
}
