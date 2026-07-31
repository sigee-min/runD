#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include "src/host/net/ready/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <unistd.h>

int RunNetLifecycleTicketCase() {
  using namespace rund::node::test_contract::net_lifecycle;

  SocketPairCleanup pair{};
  TEST_ASSERT(MakeSocketPair(pair));
  rund::net::Socket writer = rund::node::test::net::admit(pair.left);
  rund::net::Socket reader = rund::node::test::net::admit(pair.right);
  TEST_ASSERT(rund::net::nonblocking(writer.view(), true));
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true));

  const std::array<std::byte, 1u> byte{std::byte{'t'}};
  std::array<std::byte, 1u> received{};
  rund::net::ReceiveResult wrong_interest{};
  rund::net::SendResult sent{};
  rund::net::SendResult consumed{};
  rund::net::ReceiveResult read{};
  rund::net::SendResult closed{};
  rund::net::CloseResult close_result{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(RunSpec(), [&] {
    auto exercise = [&]() -> rund::task::Task<void> {
      auto wrong = co_await rund::net::ready::write(writer.view());
      wrong_interest = rund::net::receive(std::move(wrong), received);

      auto one_shot = co_await rund::net::ready::write(writer.view());
      sent = rund::net::send(std::move(one_shot), byte);
      consumed = rund::net::send(std::move(one_shot), byte);

      auto readable = co_await rund::net::ready::read(reader.view());
      read = rund::net::receive(std::move(readable), received);

      auto stale = co_await rund::net::ready::write(writer.view());
      close_result = writer.close();
      closed = rund::net::send(std::move(stale), byte);
    };
    const rund::task::Handle task =
        rund::task::spawn("net-ticket-contract", exercise());
    joined = rund::task::join(task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(!wrong_interest);
  TEST_ASSERT(wrong_interest.code() ==
              rund::ReasonCode::NetTicketInterestMismatch);
  TEST_ASSERT(sent);
  TEST_ASSERT(sent.bytes == 1);
  TEST_ASSERT(!consumed);
  TEST_ASSERT(consumed.code() == rund::ReasonCode::NetTicketConsumed);
  TEST_ASSERT(read);
  TEST_ASSERT(read.bytes == 1);
  TEST_ASSERT(received == byte);
  TEST_ASSERT(close_result);
  TEST_ASSERT(!closed);
  TEST_ASSERT(closed.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(report.tasks().network().send_calls() == 1u);
  TEST_ASSERT(report.tasks().network().recv_calls() == 1u);

  SocketPairCleanup warm_pair{};
  TEST_ASSERT(MakeSocketPair(warm_pair));
  rund::net::Socket warm_writer = rund::node::test::net::admit(warm_pair.left);
  rund::net::Socket warm_reader = rund::node::test::net::admit(warm_pair.right);
  TEST_ASSERT(rund::net::nonblocking(warm_writer.view(), true));
  TEST_ASSERT(rund::net::nonblocking(warm_reader.view(), true));
  auto prepared_ticket = rund::node::test::net::ticket(
      warm_writer.view(), rund::net::ready::Interest::Writable);
  const rund::net::ready::detail::Claim prepared_claim =
      rund::net::ready::detail::claim(std::move(prepared_ticket),
                                      rund::net::ready::Interest::Writable);
  TEST_ASSERT(prepared_claim);
  rund::net::ready::detail::Operation prepared_operation =
      rund::net::ready::detail::prepare(prepared_claim);
  TEST_ASSERT(prepared_operation);
  TEST_ASSERT(prepared_operation.native() ==
              rund::node::test::net::native(warm_writer));
  TEST_ASSERT(prepared_operation.id() == warm_writer.id());
  std::array<std::byte, 1024u> warm_received{};
  runtime_task_allocation::Start();
  for (std::uint32_t index = 0u; index < warm_received.size(); ++index) {
    auto writable = rund::node::test::net::ticket(
        warm_writer.view(), rund::net::ready::Interest::Writable);
    const rund::net::SendResult warm =
        rund::net::send(std::move(writable), byte);
    TEST_ASSERT(warm);
    const rund::net::SendResult rejected =
        rund::net::send(std::move(writable), byte);
    TEST_ASSERT(!rejected);
    TEST_ASSERT(rejected.code() == rund::ReasonCode::NetTicketConsumed);
    const rund::net::ReceiveResult drained = rund::net::direct::receive(
        warm_reader.view(), std::span<std::byte>{&warm_received[index], 1u});
    TEST_ASSERT(drained);
    TEST_ASSERT(drained.bytes == 1);
  }
  runtime_task_allocation::Stop();
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  for (const std::byte value : warm_received) {
    TEST_ASSERT(value == byte[0]);
  }
  std::array<std::byte, 1u> warm_extra{};
  const rund::net::ReceiveResult no_extra = rund::net::direct::receive(
      warm_reader.view(), std::span<std::byte>{warm_extra});
  TEST_ASSERT(!no_extra);
  TEST_ASSERT(no_extra.code() == rund::ReasonCode::IoWouldBlock);

  SocketPairCleanup old_pair{};
  TEST_ASSERT(MakeSocketPair(old_pair));
  rund::net::Socket old = rund::node::test::net::admit(old_pair.left);
  const int reused = rund::node::test::net::native(old);
  const rund::net::SocketView stale_view = old.view();
  auto stale_ticket = rund::node::test::net::ticket(
      stale_view, rund::net::ready::Interest::Writable);
  TEST_ASSERT(old.close());

  SocketPairCleanup replacement{};
  TEST_ASSERT(MakeSocketPair(replacement));
  if (replacement.left != reused) {
    TEST_ASSERT(::dup2(replacement.left, reused) == reused);
    static_cast<void>(::close(replacement.left));
    replacement.left = reused;
  }
  rund::net::Socket current = rund::node::test::net::admit(replacement.left);
  rund::net::Socket current_reader =
      rund::node::test::net::admit(replacement.right);
  TEST_ASSERT(current);
  TEST_ASSERT(current_reader);
  TEST_ASSERT(rund::net::detail::SocketAccess::slot(stale_view) ==
              rund::net::detail::SocketAccess::slot(current.view()));
  TEST_ASSERT(rund::node::test::net::generation(current) !=
              rund::net::detail::SocketAccess::generation(stale_view));

  const rund::net::SendResult stale_send =
      rund::net::send(std::move(stale_ticket), byte);
  TEST_ASSERT(!stale_send);
  TEST_ASSERT(stale_send.code() == rund::ReasonCode::IoFdInvalid);
  auto current_ticket = rund::node::test::net::ticket(
      current.view(), rund::net::ready::Interest::Writable);
  const rund::net::SendResult current_send =
      rund::net::send(std::move(current_ticket), byte);
  TEST_ASSERT(current_send);
  std::array<std::byte, 2u> current_received{};
  const rund::net::ReceiveResult current_read = rund::net::direct::receive(
      current_reader.view(), std::span<std::byte>{current_received});
  TEST_ASSERT(current_read);
  TEST_ASSERT(current_read.bytes == 1);
  TEST_ASSERT(current_received[0] == byte[0]);
  return 0;
}
