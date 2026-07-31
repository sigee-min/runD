#include "src/host/net/test/ticket.hpp"
#include "../local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetRejectsStaleMemberWait() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  std::array<std::byte, 1u> byte{std::byte{'s'}};
  READY_SET_ASSERT(rund::net::send(rund::node::test::net::ticket(
                                       writers[1u].view(),
                                       rund::net::ready::Interest::Writable),
                                   std::span<const std::byte>{byte})
                       .ok());

  rund::net::ready::Status created{};
  rund::net::ready::Status first_add{};
  rund::net::ready::Status second_add{};
  rund::net::CloseResult closed{};
  rund::net::ready::many::Result stale_wait{};
  rund::net::ready::Status destroyed{};
  std::array<rund::net::ready::Event, 2u> events{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      created =
          rund::net::ready::create(rund::net::ready::Config{.max_members = 2u});
      if (!created.ok()) {
        co_return;
      }
      first_add = rund::net::ready::add(
          created.set, ready_sets::ReadableRequest(readers[0u].view()));
      second_add = rund::net::ready::add(
          created.set, ready_sets::ReadableRequest(readers[1u].view()));
      closed = readers[0u].close();
      stale_wait = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events});
      destroyed = rund::net::ready::destroy(created.set);
    };
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-stale-wait", wait());
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(first_add.ok());
  READY_SET_ASSERT(second_add.ok());
  READY_SET_ASSERT(closed.ok());
  READY_SET_ASSERT(!stale_wait.ok());
  READY_SET_ASSERT(stale_wait.code() == rund::ReasonCode::IoFdInvalid);
  READY_SET_ASSERT(stale_wait.events == 1u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[0u].ticket.code() == rund::ReasonCode::IoFdInvalid);
  READY_SET_ASSERT(events[0u].ticket.id() == readers[0u].view().id());
  READY_SET_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
