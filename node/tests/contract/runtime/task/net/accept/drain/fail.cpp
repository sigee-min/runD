#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/accept.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <sys/socket.h>
#include <unistd.h>

int RunAcceptDrainNonWouldBlockFailureCase() {
  int fds[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  AcceptDrainSocketCleanup left{fds[0]};
  AcceptDrainSocketCleanup right{fds[1]};
  rund::net::Socket left_socket = rund::node::test::net::admit(left.fd);
  rund::net::Socket right_socket = rund::node::test::net::admit(right.fd);
  TEST_ASSERT(rund::net::nonblocking(left_socket.view(), true).ok());
  TEST_ASSERT(rund::net::nonblocking(right_socket.view(), true).ok());
  const std::array<std::byte, 1u> byte{std::byte{'a'}};
  TEST_ASSERT(rund::net::direct::send(right_socket.view(), byte).ok());

  std::uint64_t callback_count = 0u;
  rund::net::accept::Drain drained{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(AcceptDrainRunSpec(), [&] {
    auto drain = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::accept::drain(
          left_socket.view(), rund::net::accept::Budget{.max_accepts = 1u},
          [&](rund::net::accept::Result &&accepted) {
            (void)accepted;
            ++callback_count;
            return true;
          });
      if (result) {
        drained = *result;
      }
    };
    const rund::task::Handle handle =
        rund::task::spawn("accept-drain-failure", drain());
    joined = rund::task::join(handle);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!drained.ok());
  TEST_ASSERT(drained.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(drained.accepts == 0u);
  TEST_ASSERT(!drained.would_block);
  TEST_ASSERT(!drained.budget_exhausted);
  TEST_ASSERT(!drained.handler_stopped);
  TEST_ASSERT(callback_count == 0u);

  auto zero_ticket = rund::node::test::net::ticket(
      left_socket.view(), rund::net::ready::Interest::Readable);
  const auto keep = [](const void *, rund::net::accept::Result &&) noexcept {
    return true;
  };
  const rund::net::accept::Drain zero = rund::net::accept::detail::drain(
      std::move(zero_ticket), rund::net::accept::Budget{.max_accepts = 0u},
      nullptr, keep);
  TEST_ASSERT(zero.ok());
  TEST_ASSERT(zero.budget_exhausted);
  TEST_ASSERT(zero_ticket.consumed());
  TEST_ASSERT(rund::net::accept::one(std::move(zero_ticket)).code() ==
              rund::ReasonCode::NetTicketConsumed);

  auto null_ticket = rund::node::test::net::ticket(
      left_socket.view(), rund::net::ready::Interest::Readable);
  const rund::net::accept::Drain null = rund::net::accept::detail::drain(
      std::move(null_ticket), rund::net::accept::Budget{.max_accepts = 1u},
      nullptr, nullptr);
  TEST_ASSERT(null.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(null_ticket.consumed());
  TEST_ASSERT(rund::net::accept::one(std::move(null_ticket)).code() ==
              rund::ReasonCode::NetTicketConsumed);
  return 0;
}
