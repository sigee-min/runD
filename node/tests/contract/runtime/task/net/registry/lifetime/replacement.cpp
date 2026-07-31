#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <unistd.h>

int RunNetRegistryLifetimeReplacementCase() {
  using rund::node::test_contract::net_registry_lifetime::ForceLeftFd;
  using rund::node::test_contract::net_registry_lifetime::MakeSocketPair;
  using rund::node::test_contract::net_registry_lifetime::SocketPair;

  SocketPair old_pair{};
  SocketPair new_pair{};
  TEST_ASSERT(MakeSocketPair(old_pair));

  rund::net::Socket stale_socket{};
  rund::net::Socket replacement_socket{};
  rund::net::ready::Ticket stale_wait{};
  rund::net::ready::Ticket replacement_wait{};
  rund::task::Status joined{};
  rund::task::Status stale_joined{};
  rund::task::Status replacement_joined{};
  bool setup_ok = true;

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        auto scenario = [&]() -> rund::task::Task<void> {
          stale_socket = rund::node::test::net::admit(old_pair.left);
          if (rund::node::test::net::generation(stale_socket) == 0u) {
            setup_ok = false;
            co_return;
          }
          const int reused_fd = rund::node::test::net::native(stale_socket);

          auto wait_stale = [&]() -> rund::task::Task<void> {
            stale_wait = co_await rund::net::ready::read(stale_socket.view());
          };
          const rund::task::Handle stale_waiter =
              rund::task::spawn("net-registry-stale-wait", wait_stale());
          (void)co_await rund::task::yield();

          static_cast<void>(::close(reused_fd));
          if (!MakeSocketPair(new_pair) || !ForceLeftFd(new_pair, reused_fd)) {
            setup_ok = false;
            co_return;
          }

          replacement_socket = rund::node::test::net::admit(new_pair.left);
          if (replacement_socket.id() != stale_socket.id() ||
              rund::node::test::net::generation(replacement_socket) == 0u ||
              rund::node::test::net::generation(replacement_socket) ==
                  rund::node::test::net::generation(stale_socket)) {
            setup_ok = false;
            co_return;
          }

          const char byte = 'r';
          if (::write(new_pair.right, &byte, 1u) != 1) {
            setup_ok = false;
            co_return;
          }
          auto wait_replacement = [&]() -> rund::task::Task<void> {
            replacement_wait =
                co_await rund::net::ready::read(replacement_socket.view());
          };
          const rund::task::Handle replacement_waiter = rund::task::spawn(
              "net-registry-replacement-wait", wait_replacement());
          stale_joined = co_await stale_waiter;
          replacement_joined = co_await replacement_waiter;
        };
        const rund::task::Handle task =
            rund::task::spawn("net-registry-replacement", scenario());
        joined = rund::task::join(task);
      });

  TEST_ASSERT(setup_ok);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(stale_joined.ok());
  TEST_ASSERT(replacement_joined.ok());
  TEST_ASSERT(replacement_wait.ok());
  TEST_ASSERT(!stale_wait.ok());
  TEST_ASSERT(stale_wait.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(report.tasks().network().admission_rejections() == 0u);
  return 0;
}
