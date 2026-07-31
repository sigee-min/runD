#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include "../local.hpp"

#include <rund/net/ready.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <chrono>
#include <span>

#include <unistd.h>

int RunNetReadyManyConnectGenerationCase() {
  std::array<SocketPairCleanup, 2u> cleanup{};
  SocketPairCleanup replacement_cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup[0u]));
  TEST_ASSERT(MakeSocketPair(cleanup[1u]));
  std::array<rund::net::ready::Request, 2u> requests{};
  std::array<rund::net::ready::Event, 2u> events{};
  rund::net::ready::many::Result stale_result{};
  rund::net::ready::Ticket new_generation_ready{};
  rund::task::Status yielded{};
  rund::task::Status joined{};
  bool setup_ok = true;
  const rund::Session::Result report =
      rund::run(ReadyManyRunSpec(4u, 8u, 16u), [&] {
        rund::net::Socket old_socket =
            rund::node::test::net::admit(cleanup[0u].left);
        rund::net::Socket sibling_socket =
            rund::node::test::net::admit(cleanup[1u].left);
        if (rund::node::test::net::generation(old_socket) == 0u ||
            rund::node::test::net::generation(sibling_socket) == 0u) {
          setup_ok = false;
          return;
        }
        if (!rund::net::nonblocking(old_socket.view(), true).ok() ||
            !rund::net::nonblocking(sibling_socket.view(), true).ok()) {
          setup_ok = false;
          return;
        }
        const int reused_fd = rund::node::test::net::native(old_socket);
        requests[0u] = rund::net::ready::Request{
            .socket = old_socket.view(),
            .interest = rund::net::ready::Interest::Readable};
        requests[1u] = rund::net::ready::Request{
            .socket = sibling_socket.view(),
            .interest = rund::net::ready::Interest::Readable};

        auto wait = [&]() -> rund::task::Task<void> {
          stale_result = co_await rund::net::ready::many::wait(
              std::span<const rund::net::ready::Request>{requests},
              std::span<rund::net::ready::Event>{events},
              std::chrono::seconds{30});
        };
        const rund::task::Handle stale_waiter =
            rund::task::spawn("net-ready-many-generation-stale", wait());

        auto replace = [&]() -> rund::task::Task<void> {
          yielded = co_await rund::task::yield();
          static_cast<void>(::close(reused_fd));
          if (!MakeSocketPair(replacement_cleanup)) {
            setup_ok = false;
            co_return;
          }
          if (replacement_cleanup.left != reused_fd) {
            if (::dup2(replacement_cleanup.left, reused_fd) != reused_fd) {
              setup_ok = false;
              co_return;
            }
            static_cast<void>(::close(replacement_cleanup.left));
            replacement_cleanup.left = reused_fd;
          }
          rund::net::Socket new_socket =
              rund::node::test::net::admit(replacement_cleanup.left);
          if (new_socket.id() != old_socket.id() ||
              rund::node::test::net::generation(new_socket) == 0u ||
              rund::node::test::net::generation(new_socket) ==
                  rund::node::test::net::generation(old_socket)) {
            setup_ok = false;
            co_return;
          }
          if (!rund::net::nonblocking(new_socket.view(), true).ok()) {
            setup_ok = false;
            co_return;
          }
          const char byte = 'g';
          if (::write(replacement_cleanup.right, &byte, 1u) != 1) {
            setup_ok = false;
            co_return;
          }
          new_generation_ready =
              co_await rund::net::ready::read(new_socket.view());
        };
        const rund::task::Handle replacer =
            rund::task::spawn("net-ready-many-generation-replacer", replace());
        joined = rund::task::join(stale_waiter, replacer);
      });

  TEST_ASSERT(setup_ok);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(yielded.ok());
  TEST_ASSERT(new_generation_ready.ok());
  TEST_ASSERT(report.tasks().reactor_waits() == 2u);
  TEST_ASSERT(!stale_result.ok());
  TEST_ASSERT(stale_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(stale_result.events == 1u);
  TEST_ASSERT(events[0u].index == 0u);
  TEST_ASSERT(events[0u].ticket.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(events[0u].ticket.id() == requests[0u].socket.id());
  TEST_ASSERT(report.tasks().failed() == 0u);
  return 0;
}
