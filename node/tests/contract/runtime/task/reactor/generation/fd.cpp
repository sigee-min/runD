#include "src/host/net/test/socket.hpp"
#include "../await.hpp"
#include "test/assert.hpp"

#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <cstdint>

#include <unistd.h>

namespace {

[[nodiscard]] bool MakePipe(int (&fds)[2]) {
  fds[0] = -1;
  fds[1] = -1;
  return ::pipe(fds) == 0;
}

void CloseFd(int &fd) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
    fd = -1;
  }
}

} // namespace

int RunRuntimeTaskReactorFdGenerationContract() {
  int first[2] = {-1, -1};
  int second[2] = {-1, -1};
  TEST_ASSERT(MakePipe(first));
  const int reused_fd = first[0];

  rund::net::ready::Ticket stale_wait{};
  rund::net::ready::Ticket new_ready{};
  rund::task::Status joined{};
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
        rund::net::Socket old_socket = rund::node::test::net::admit(first[0]);
        if (!old_socket ||
            rund::node::test::net::generation(old_socket) == 0u) {
          setup_ok = false;
          return;
        }
        if (!rund::net::nonblocking(old_socket.view(), true).ok()) {
          setup_ok = false;
          return;
        }
        const std::uint64_t old_id = old_socket.id();
        const std::uint64_t old_generation =
            rund::node::test::net::generation(old_socket);
        const rund::task::Handle waiter =
            rund::task::spawn("stale-generation-waiter",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  old_socket.view(), &stale_wait));
        rund::task::Handle new_waiter{};
        rund::net::Socket new_socket{};
        auto replace = [&]() -> rund::task::Task<void> {
          (void)co_await rund::task::yield();
          if (!old_socket.close().ok()) {
            setup_ok = false;
            co_return;
          }
          if (!MakePipe(second)) {
            setup_ok = false;
            co_return;
          }
          if (second[0] != reused_fd) {
            if (::dup2(second[0], reused_fd) != reused_fd) {
              setup_ok = false;
              co_return;
            }
            CloseFd(second[0]);
            second[0] = reused_fd;
          }
          new_socket = rund::node::test::net::admit(second[0]);
          if (!new_socket) {
            setup_ok = false;
            co_return;
          }
          if (!rund::net::nonblocking(new_socket.view(), true).ok()) {
            setup_ok = false;
            co_return;
          }
          if (new_socket.id() != old_id ||
              rund::node::test::net::generation(new_socket) == 0u ||
              rund::node::test::net::generation(new_socket) == old_generation) {
            setup_ok = false;
            co_return;
          }
          const char byte = 'g';
          if (::write(second[1], &byte, 1u) != 1) {
            setup_ok = false;
            co_return;
          }
          new_waiter = rund::task::spawn(
              "new-generation-waiter",
              rund::node::test_contract::reactor::AwaitReadable(
                  new_socket.view(), &new_ready));
        };
        const rund::task::Handle replacer =
            rund::task::spawn("replace-generation", replace());
        joined = rund::task::join(waiter, replacer);
        if (joined.ok()) {
          joined = rund::task::join(new_waiter);
        }
      });

  CloseFd(first[0]);
  CloseFd(first[1]);
  CloseFd(second[0]);
  CloseFd(second[1]);

  TEST_ASSERT(setup_ok);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(new_ready.ok());
  TEST_ASSERT(!stale_wait.ok());
  TEST_ASSERT(stale_wait.code() == rund::ReasonCode::IoFdInvalid);
  return 0;
}
