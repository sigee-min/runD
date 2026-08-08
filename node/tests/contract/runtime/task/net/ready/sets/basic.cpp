#include "local.hpp"
#include "src/host/net/test/ticket.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "src/runtime/task/scheduler/reactor/invariants.hpp"

namespace rund::node::test_contract {

bool NetReadySetCreateAddWaitDestroy() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  std::array<std::byte, 1u> byte{std::byte{'r'}};
  READY_SET_ASSERT(rund::net::send(rund::node::test::net::ticket(
                                       writers[1u].view(),
                                       rund::net::ready::Interest::Writable),
                                   std::span<const std::byte>{byte})
                       .ok());
  READY_SET_ASSERT(rund::net::send(rund::node::test::net::ticket(
                                       writers[0u].view(),
                                       rund::net::ready::Interest::Writable),
                                   std::span<const std::byte>{byte})
                       .ok());

  rund::net::ready::Status created{};
  rund::net::ready::Status first_add{};
  rund::net::ready::Status second_add{};
  rund::net::ready::many::Result ready{};
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
      ready = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events});
      destroyed = rund::net::ready::destroy(created.set);
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-set-create-add-wait-destroy", wait());
    joined = rund::task::join(waiter);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(created.set.id != 0u);
  READY_SET_ASSERT(created.set.generation != 0u);
  READY_SET_ASSERT(first_add.ok());
  READY_SET_ASSERT(second_add.ok());
  READY_SET_ASSERT(ready.ok());
  READY_SET_ASSERT(ready.events == 2u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[1u].index == 1u);
  READY_SET_ASSERT(events[0u].ticket.id() == readers[0u].id());
  READY_SET_ASSERT(events[1u].ticket.id() == readers[1u].id());
  READY_SET_ASSERT(destroyed.ok());
  READY_SET_ASSERT(report.tasks().reactor().ready_set_members() == 2u);
  READY_SET_ASSERT(report.tasks().reactor().ready_set_ready_events() == 2u);
  READY_SET_ASSERT(report.tasks().reactor().ready_set_invalidations() == 2u);
  READY_SET_ASSERT(report.tasks().resources().live_ready_set_members() == 0u);
  return true;
}

bool NetReadySetParkWake() {
  ready_sets::SocketPairCleanup cleanup{};
  rund::net::Socket reader{};
  rund::net::Socket writer{};
  READY_SET_ASSERT(ready_sets::PrepareSocketPair(cleanup, reader, writer));

  rund::net::ready::Status created{};
  rund::net::ready::Status added{};
  rund::net::ready::Status destroyed{};
  rund::net::ready::many::Result ready{};
  rund::net::SendResult sent{};
  rund::task::Status joined{};
  rund::node::ReactorInvariantSnapshot before_wait{};
  rund::node::ReactorInvariantSnapshot after_wait{};
  std::array<rund::net::ready::Event, 1u> events{};
  const std::array<std::byte, 1u> byte{std::byte{'p'}};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    created =
        rund::net::ready::create(rund::net::ready::Config{.max_members = 1u});
    if (!created.ok()) {
      return;
    }
    added = rund::net::ready::add(created.set,
                                  ready_sets::ReadableRequest(reader.view()));
    if (!added.ok()) {
      return;
    }
    before_wait = rund::node::ValidateReactorCleanupInvariantsForTest();
    auto wait = [&]() -> rund::task::Task<void> {
      ready = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events},
          std::chrono::seconds{5});
    };
    auto write = [&]() -> rund::task::Task<void> {
      (void)co_await rund::task::sleep(std::chrono::milliseconds{1});
      sent = rund::net::send(
          rund::node::test::net::ticket(writer.view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-set-park-wait", wait());
    const rund::task::Handle sender =
        rund::task::spawn("net-ready-set-park-send", write());
    joined = rund::task::join(waiter, sender);
    after_wait = rund::node::ValidateReactorCleanupInvariantsForTest();
    destroyed = rund::net::ready::destroy(created.set);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(added.ok());
  READY_SET_ASSERT(sent.ok() && sent.bytes == 1);
  READY_SET_ASSERT(ready.ok() && !ready.timed_out());
  READY_SET_ASSERT(ready.events == 1u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[0u].ticket.id() == reader.id());
  READY_SET_ASSERT(destroyed.ok());
  READY_SET_ASSERT(before_wait.ok && after_wait.ok);
  READY_SET_ASSERT(after_wait.many_request_copies ==
                   before_wait.many_request_copies + 1u);
  READY_SET_ASSERT(after_wait.many_storage_growths ==
                   before_wait.many_storage_growths);
  READY_SET_ASSERT(report.tasks().coroutine_parks() >= 1u);
  READY_SET_ASSERT(report.tasks().reactor_waits() == 1u);
  return true;
}

bool NetReadySetsWakeInCausalOrder() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  std::array<rund::net::ready::Status, 2u> created{};
  std::array<rund::net::ready::Status, 2u> added{};
  std::array<rund::net::ready::Status, 2u> destroyed{};
  std::array<rund::net::ready::many::Result, 2u> ready{};
  std::array<rund::net::SendResult, 2u> sent{};
  std::array<std::array<rund::net::ready::Event, 1u>, 2u> events{};
  std::array<std::uint32_t, 2u> wake_order{};
  std::uint32_t wake_count = 0u;
  rund::task::Status joined{};
  rund::task::Status causal_join{};
  const std::array<std::byte, 1u> byte{std::byte{'o'}};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    for (std::size_t index = 0u; index < created.size(); ++index) {
      created[index] =
          rund::net::ready::create(rund::net::ready::Config{.max_members = 1u});
      if (!created[index].ok()) {
        return;
      }
      added[index] = rund::net::ready::add(
          created[index].set,
          ready_sets::ReadableRequest(readers[index].view()));
      if (!added[index].ok()) {
        return;
      }
    }
    auto wait_first = [&]() -> rund::task::Task<void> {
      ready[0u] = co_await rund::net::ready::many::wait(
          created[0u].set, std::span<rund::net::ready::Event>{events[0u]},
          std::chrono::seconds{5});
      wake_order[wake_count++] = 0u;
    };
    auto wait_second = [&]() -> rund::task::Task<void> {
      ready[1u] = co_await rund::net::ready::many::wait(
          created[1u].set, std::span<rund::net::ready::Event>{events[1u]},
          std::chrono::seconds{5});
      wake_order[wake_count++] = 1u;
    };
    const rund::task::Handle first =
        rund::task::spawn("net-ready-set-order-first", wait_first());
    const rund::task::Handle second =
        rund::task::spawn("net-ready-set-order-second", wait_second());
    auto write_causal = [&, second]() -> rund::task::Task<void> {
      sent[1u] = rund::net::send(
          rund::node::test::net::ticket(writers[1u].view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
      causal_join = co_await second;
      sent[0u] = rund::net::send(
          rund::node::test::net::ticket(writers[0u].view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
    };
    const rund::task::Handle sender =
        rund::task::spawn("net-ready-set-order-send", write_causal());
    joined = rund::task::join(first, sender);
    for (std::size_t index = 0u; index < destroyed.size(); ++index) {
      destroyed[index] = rund::net::ready::destroy(created[index].set);
    }
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(causal_join.ok());
  READY_SET_ASSERT(wake_count == 2u);
  READY_SET_ASSERT(wake_order[0u] == 1u && wake_order[1u] == 0u);
  for (std::size_t index = 0u; index < ready.size(); ++index) {
    READY_SET_ASSERT(created[index].ok() && added[index].ok());
    READY_SET_ASSERT(sent[index].ok() && sent[index].bytes == 1);
    READY_SET_ASSERT(ready[index].ok() && ready[index].events == 1u);
    READY_SET_ASSERT(events[index][0u].index == 0u);
    READY_SET_ASSERT(events[index][0u].ticket.id() == readers[index].id());
    READY_SET_ASSERT(destroyed[index].ok());
  }
  READY_SET_ASSERT(report.tasks().reactor_waits() == 2u);
  return true;
}

} // namespace rund::node::test_contract
