#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "src/runtime/task/scheduler/reactor/invariants.hpp"

namespace rund::node::test_contract {

bool NetReadySetClearResetsInsertionIndex() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  std::array<std::byte, 1u> byte{std::byte{'i'}};
  READY_SET_ASSERT(rund::net::send(rund::node::test::net::ticket(
                                       writers[1u].view(),
                                       rund::net::ready::Interest::Writable),
                                   std::span<const std::byte>{byte})
                       .ok());

  rund::net::ready::Status created{};
  rund::net::ready::Status first_add{};
  rund::net::ready::Status second_add{};
  rund::net::ready::Status cleared{};
  rund::net::ready::Status after_clear_add{};
  rund::net::ready::many::Result ready{};
  rund::net::ready::Status destroyed{};
  std::array<rund::net::ready::Event, 1u> events{};
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
      cleared = rund::net::ready::clear(created.set);
      after_clear_add = rund::net::ready::add(
          created.set, ready_sets::ReadableRequest(readers[1u].view()));
      ready = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events});
      destroyed = rund::net::ready::destroy(created.set);
    };
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-clear-resets-index", wait());
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(first_add.ok());
  READY_SET_ASSERT(second_add.ok());
  READY_SET_ASSERT(cleared.ok());
  READY_SET_ASSERT(after_clear_add.ok());
  READY_SET_ASSERT(ready.ok());
  READY_SET_ASSERT(ready.events == 1u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[0u].ticket.id() == readers[1u].id());
  READY_SET_ASSERT(destroyed.ok());
  return true;
}

bool NetReadySetRemovalHoleKeepsInsertionIndex() {
  std::array<ready_sets::SocketPairCleanup, 3u> cleanup{};
  std::array<rund::net::Socket, 3u> readers{};
  std::array<rund::net::Socket, 3u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  rund::net::ready::Status created{};
  std::array<rund::net::ready::Status, 3u> added{};
  rund::net::ready::Status removed{};
  rund::net::ready::Status destroyed{};
  rund::net::ready::many::Result ready{};
  rund::net::SendResult sent{};
  rund::task::Status joined{};
  std::array<rund::net::ready::Event, 1u> events{};
  const std::array<std::byte, 1u> byte{std::byte{'h'}};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    created =
        rund::net::ready::create(rund::net::ready::Config{.max_members = 3u});
    if (!created.ok()) {
      return;
    }
    for (std::size_t index = 0u; index < added.size(); ++index) {
      added[index] = rund::net::ready::add(
          created.set, ready_sets::ReadableRequest(readers[index].view()));
    }
    removed = rund::net::ready::remove(
        created.set, ready_sets::ReadableRequest(readers[1u].view()));
    auto wait = [&]() -> rund::task::Task<void> {
      ready = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events},
          std::chrono::seconds{5});
    };
    auto write = [&]() -> rund::task::Task<void> {
      (void)co_await rund::task::sleep(std::chrono::milliseconds{1});
      sent = rund::net::send(
          rund::node::test::net::ticket(writers[2u].view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-set-hole-wait", wait());
    const rund::task::Handle sender =
        rund::task::spawn("net-ready-set-hole-send", write());
    joined = rund::task::join(waiter, sender);
    destroyed = rund::net::ready::destroy(created.set);
  });

  READY_SET_ASSERT(report.ok() && joined.ok());
  READY_SET_ASSERT(created.ok() && removed.ok() && destroyed.ok());
  for (const auto &result : added) {
    READY_SET_ASSERT(result.ok());
  }
  READY_SET_ASSERT(sent.ok() && sent.bytes == 1);
  READY_SET_ASSERT(ready.ok() && ready.events == 1u);
  READY_SET_ASSERT(events[0u].index == 2u);
  READY_SET_ASSERT(events[0u].ticket.id() == readers[2u].id());
  return true;
}

bool NetReadySetWaitUsesMembershipSnapshot() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  rund::net::ready::Status created{};
  rund::net::ready::Status first_add{};
  rund::net::ready::Status removed{};
  rund::net::ready::Status second_add{};
  rund::net::ready::Status destroyed{};
  rund::net::ready::many::Result ready{};
  rund::net::SendResult sent{};
  rund::task::Status joined{};
  std::array<rund::net::ready::Event, 1u> events{};
  const std::array<std::byte, 1u> byte{std::byte{'s'}};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    created =
        rund::net::ready::create(rund::net::ready::Config{.max_members = 2u});
    if (!created.ok()) {
      return;
    }
    first_add = rund::net::ready::add(
        created.set, ready_sets::ReadableRequest(readers[0u].view()));
    auto wait = [&]() -> rund::task::Task<void> {
      ready = co_await rund::net::ready::many::wait(
          created.set, std::span<rund::net::ready::Event>{events},
          std::chrono::seconds{5});
    };
    auto mutate = [&]() -> rund::task::Task<void> {
      (void)co_await rund::task::sleep(std::chrono::milliseconds{1});
      removed = rund::net::ready::remove(
          created.set, ready_sets::ReadableRequest(readers[0u].view()));
      second_add = rund::net::ready::add(
          created.set, ready_sets::ReadableRequest(readers[1u].view()));
      sent = rund::net::send(
          rund::node::test::net::ticket(writers[0u].view(),
                                        rund::net::ready::Interest::Writable),
          std::span<const std::byte>{byte});
    };
    const rund::task::Handle waiter =
        rund::task::spawn("net-ready-set-snapshot-wait", wait());
    const rund::task::Handle mutator =
        rund::task::spawn("net-ready-set-snapshot-mutate", mutate());
    joined = rund::task::join(waiter, mutator);
    destroyed = rund::net::ready::destroy(created.set);
  });

  READY_SET_ASSERT(report.ok() && joined.ok());
  READY_SET_ASSERT(created.ok() && first_add.ok());
  READY_SET_ASSERT(removed.ok() && second_add.ok() && destroyed.ok());
  READY_SET_ASSERT(sent.ok() && sent.bytes == 1);
  READY_SET_ASSERT(ready.ok() && ready.events == 1u);
  READY_SET_ASSERT(events[0u].index == 0u);
  READY_SET_ASSERT(events[0u].ticket.id() == readers[0u].id());
  return true;
}

bool NetReadySetChurnStoragePlateaus() {
  ready_sets::SocketPairCleanup cleanup{};
  rund::net::Socket reader{};
  rund::net::Socket writer{};
  READY_SET_ASSERT(ready_sets::PrepareSocketPair(cleanup, reader, writer));

  rund::net::ready::Status created{};
  rund::net::ready::Status destroyed{};
  rund::node::ReactorInvariantSnapshot before{};
  rund::node::ReactorInvariantSnapshot after{};
  rund::task::Status joined{};
  bool churn_ok = true;

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-churn", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          if (!created.ok()) {
            churn_ok = false;
            return;
          }
          before = rund::node::ValidateReactorCleanupInvariantsForTest();
          const rund::net::ready::Request request =
              ready_sets::ReadableRequest(reader.view());
          for (std::uint32_t iteration = 0u; iteration < 100'000u;
               ++iteration) {
            if (!rund::net::ready::add(created.set, request).ok() ||
                !rund::net::ready::remove(created.set, request).ok()) {
              churn_ok = false;
              return;
            }
          }
          after = rund::node::ValidateReactorCleanupInvariantsForTest();
          destroyed = rund::net::ready::destroy(created.set);
        });
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok() && joined.ok() && churn_ok);
  READY_SET_ASSERT(created.ok() && destroyed.ok());
  READY_SET_ASSERT(before.ok && after.ok);
  READY_SET_ASSERT(after.ready_set_member_storage == 0u);
  READY_SET_ASSERT(after.ready_set_member_capacity <= 1u);
  READY_SET_ASSERT(after.ready_set_storage_growths ==
                   before.ready_set_storage_growths);
  return true;
}

} // namespace rund::node::test_contract
