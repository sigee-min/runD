#include "../../local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetRejectsStaleWaitMember() {
  ready_sets::SocketPairCleanup cleanup{};
  rund::net::Socket reader{};
  rund::net::Socket writer{};
  READY_SET_ASSERT(ready_sets::PrepareSocketPair(cleanup, reader, writer));

  rund::net::ready::Status created{};
  rund::net::ready::Status added{};
  rund::net::CloseResult closed{};
  rund::net::ready::many::Result stale_wait{};
  rund::net::ready::Status destroyed{};
  std::array<rund::net::ready::Event, 1u> events{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    auto wait = [&]() -> rund::task::Task<void> {
      created =
          rund::net::ready::create(rund::net::ready::Config{.max_members = 1u});
      if (!created.ok()) {
        co_return;
      }
      added = rund::net::ready::add(created.set,
                                    ready_sets::ReadableRequest(reader.view()));
      closed = reader.close();
      stale_wait = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events});
      destroyed = rund::net::ready::destroy(created.set);
    };
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-stale-member", wait());
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(added.ok());
  READY_SET_ASSERT(closed.ok());
  READY_SET_ASSERT(!stale_wait.ok());
  READY_SET_ASSERT(stale_wait.code() == rund::ReasonCode::IoFdInvalid);
  READY_SET_ASSERT(stale_wait.events == 1u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[0u].ticket.code() == rund::ReasonCode::IoFdInvalid);
  READY_SET_ASSERT(events[0u].ticket.id() == reader.view().id());
  READY_SET_ASSERT(destroyed.ok());
  READY_SET_ASSERT(report.tasks().reactor().ready_set_ready_events() == 1u);
  READY_SET_ASSERT(report.tasks().reactor().ready_set_invalidations() >= 1u);
  return true;
}

} // namespace rund::node::test_contract
