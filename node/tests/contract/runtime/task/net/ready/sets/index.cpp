#include "local.hpp"
#include "src/host/net/operation.hpp"
#include "src/host/net/test/ticket.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "src/runtime/task/scheduler/reactor/invariants.hpp"
#include "src/runtime/task/scheduler/reactor/ready/set/identity.hpp"
#include "src/runtime/task/scheduler/reactor/ready/set/store.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
#include <type_traits>

static_assert(sizeof(rund::net::ready::Set) == 16u);
static_assert(std::is_standard_layout_v<rund::net::ready::Set>);
static_assert(std::is_trivially_copyable_v<rund::net::ready::Set>);

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
  READY_SET_ASSERT(report.tasks().reactor().ready_set_members() == 3u);
  READY_SET_ASSERT(report.tasks().reactor().ready_set_members_added() == 3u);
  READY_SET_ASSERT(report.tasks().reactor().ready_set_members_removed() == 1u);
  READY_SET_ASSERT(report.tasks().resources().live_ready_set_members() == 0u);
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

bool NetReadySetIdentityTransitionsAreClosed() {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();

  ReactorReadySetIdentityOwner exhausted{maximum};
  ReactorReadySetIdentityState unchanged{};
  const ::rund::net::ready::Set sentinel{.id = 71u, .generation = 73u};
  ::rund::net::ready::many::Wait deferred =
      ::rund::net::ready::many::detail::Access::Defer({}, {}, {}, 0, false, {},
                                                      sentinel);
  ::rund::net::ready::many::Wait moved{std::move(deferred)};
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::same(
      ::rund::net::ready::many::detail::Access::Snapshot(moved).ready_set,
      sentinel));
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::empty(
      ::rund::net::ready::many::detail::Access::Snapshot(deferred).ready_set));
  ::rund::net::ready::Set output = sentinel;
  READY_SET_ASSERT(!exhausted.activate(unchanged, &output));
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::empty(unchanged.handle));
  READY_SET_ASSERT(!unchanged.live);
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::same(output, sentinel));

  ReactorReadySetIdentityOwner last_id{maximum - 1u};
  ReactorReadySetIdentityState last_identity{};
  ::rund::net::ready::Set last_handle{};
  READY_SET_ASSERT(last_id.activate(last_identity, &last_handle));
  READY_SET_ASSERT(last_handle.id == maximum - 1u &&
                   last_handle.generation == 1u);
  ReactorReadySetIdentityState after_last{};
  output = sentinel;
  READY_SET_ASSERT(!last_id.activate(after_last, &output));
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::empty(after_last.handle));
  READY_SET_ASSERT(!after_last.live);
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::same(output, sentinel));

  ReactorReadySetIdentityState reusable{
      .handle = {.id = 17u, .generation = 2u},
      .live = false,
  };
  output = {};
  READY_SET_ASSERT(exhausted.activate(reusable, &output));
  READY_SET_ASSERT(output.id == 17u && output.generation == 3u);
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::matches(reusable, output));

  ReactorReadySetIdentityState maximum_live{
      .handle = {.id = 23u, .generation = maximum},
      .live = true,
  };
  ::rund::net::ready::Set maximum_tombstone{};
  READY_SET_ASSERT(
      exhausted.retire(maximum_live, maximum_live.handle, &maximum_tombstone));
  READY_SET_ASSERT(maximum_tombstone.id == 23u &&
                   maximum_tombstone.generation == maximum);
  READY_SET_ASSERT(!maximum_live.live);
  READY_SET_ASSERT(
      !ReactorReadySetIdentityOwner::matches(maximum_live, maximum_tombstone));

  output = sentinel;
  READY_SET_ASSERT(!exhausted.activate(maximum_live, &output));
  READY_SET_ASSERT(!maximum_live.live);
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::same(maximum_live.handle,
                                                      maximum_tombstone));
  READY_SET_ASSERT(ReactorReadySetIdentityOwner::same(output, sentinel));

  ReactorReadySetIdentityOwner rekey{41u};
  ::rund::net::ready::Set rekeyed{};
  READY_SET_ASSERT(rekey.activate(maximum_live, &rekeyed));
  READY_SET_ASSERT(rekeyed.id == 41u && rekeyed.generation == 1u);
  READY_SET_ASSERT(
      ReactorReadySetIdentityOwner::matches(maximum_live, rekeyed));

  std::vector<ReactorReadySet> slots(2u);
  slots[0u].identity = ReactorReadySetIdentityState{
      .handle = {.id = 29u, .generation = maximum},
      .live = false,
  };
  slots[1u].identity = ReactorReadySetIdentityState{
      .handle = {.id = 31u, .generation = 8u},
      .live = false,
  };
  ReactorReadySet *const selected = ReactorReadySetSelectActivationSlot(slots);
  READY_SET_ASSERT(selected == &slots[1u]);
  ::rund::net::ready::Set selected_handle{};
  READY_SET_ASSERT(exhausted.activate(selected->identity, &selected_handle));
  READY_SET_ASSERT(selected_handle.id == 31u &&
                   selected_handle.generation == 9u);

  constexpr std::size_t issue_count = 1024u;
  constexpr std::size_t thread_count = 4u;
  static_assert(issue_count % thread_count == 0u);
  ReactorReadySetIdentityOwner concurrent{};
  std::array<ReactorReadySetIdentityState, issue_count> identities{};
  std::array<::rund::net::ready::Set, issue_count> handles{};
  std::array<std::thread, thread_count> threads{};
  std::atomic<bool> issued{true};
  for (std::size_t thread = 0u; thread < thread_count; ++thread) {
    threads[thread] = std::thread{[&, thread] {
      const std::size_t begin = thread * issue_count / thread_count;
      const std::size_t end = begin + issue_count / thread_count;
      for (std::size_t index = begin; index < end; ++index) {
        if (!concurrent.activate(identities[index], &handles[index])) {
          issued.store(false, std::memory_order_relaxed);
          return;
        }
      }
    }};
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  READY_SET_ASSERT(issued.load(std::memory_order_relaxed));
  std::sort(
      handles.begin(), handles.end(),
      [](const auto left, const auto right) { return left.id < right.id; });
  for (std::size_t index = 0u; index < handles.size(); ++index) {
    READY_SET_ASSERT(handles[index].id == index + 1u);
    READY_SET_ASSERT(handles[index].id != maximum);
    READY_SET_ASSERT(handles[index].generation == 1u);
  }
  return true;
}

bool NetReadySetCapabilitiesDoNotAlias() {
  const auto deterministic_run = [](::rund::net::ready::Status &created,
                                    ::rund::net::ready::Status &destroyed) {
    return ::rund::run(ready_sets::RunSpec(), [&] {
      created = ::rund::net::ready::create(
          ::rund::net::ready::Config{.max_members = 1u});
      if (created.ok()) {
        destroyed = ::rund::net::ready::destroy(created.set);
      }
    });
  };

  ::rund::net::ready::Status deterministic_first{};
  ::rund::net::ready::Status deterministic_first_destroyed{};
  const ::rund::Session::Result first_trace =
      deterministic_run(deterministic_first, deterministic_first_destroyed);
  READY_SET_ASSERT(first_trace.ok());
  READY_SET_ASSERT(deterministic_first.ok());
  READY_SET_ASSERT(deterministic_first_destroyed.ok());

  ::rund::net::ready::Status first_session{};
  const ::rund::Session::Result first_report =
      ::rund::run(ready_sets::RunSpec(), [&] {
        first_session = ::rund::net::ready::create(
            ::rund::net::ready::Config{.max_members = 1u});
      });
  READY_SET_ASSERT(first_report.ok() && first_session.ok());

  ::rund::net::ready::Status second_session{};
  ::rund::net::ready::Status foreign_clear{};
  ::rund::net::ready::Status second_destroyed{};
  const ::rund::Session::Result second_report =
      ::rund::run(ready_sets::RunSpec(), [&] {
        second_session = ::rund::net::ready::create(
            ::rund::net::ready::Config{.max_members = 1u});
        foreign_clear = ::rund::net::ready::clear(first_session.set);
        if (second_session.ok()) {
          second_destroyed = ::rund::net::ready::destroy(second_session.set);
        }
      });
  READY_SET_ASSERT(second_report.ok() && second_session.ok());
  READY_SET_ASSERT(!foreign_clear.ok());
  READY_SET_ASSERT(foreign_clear.code() == ::rund::ReasonCode::TaskInvalid);
  READY_SET_ASSERT(second_destroyed.ok());
  READY_SET_ASSERT(!ReactorReadySetIdentityOwner::same(first_session.set,
                                                       second_session.set));

  ::rund::Session persistent{};
  READY_SET_ASSERT(persistent.open(ready_sets::RunSpec()).ok());
  ::rund::net::ready::Status persistent_set{};
  const ::rund::Session::Result persistent_first = persistent.scope([&] {
    persistent_set = ::rund::net::ready::create(
        ::rund::net::ready::Config{.max_members = 1u});
  });
  READY_SET_ASSERT(persistent_first.ok() && persistent_set.ok());
  ::rund::net::ready::Status cross_scope_clear{};
  ::rund::net::ready::Status persistent_destroyed{};
  const ::rund::Session::Result persistent_second = persistent.scope([&] {
    cross_scope_clear = ::rund::net::ready::clear(persistent_set.set);
    persistent_destroyed = ::rund::net::ready::destroy(persistent_set.set);
  });
  READY_SET_ASSERT(persistent_second.ok());
  READY_SET_ASSERT(cross_scope_clear.ok() && persistent_destroyed.ok());
  READY_SET_ASSERT(persistent.close().ok());

  ::rund::net::ready::Status initial{};
  ::rund::net::ready::Status initial_destroyed{};
  ::rund::net::ready::Status replacement{};
  ::rund::net::ready::Status initial_stale{};
  ::rund::net::ready::Status tombstone_stale{};
  ::rund::net::ready::Status replacement_clear{};
  ::rund::net::ready::Status replacement_destroyed{};
  const ::rund::Session::Result reuse_report =
      ::rund::run(ready_sets::Config(2u, 2u, 4u, 1u, 1u), [&] {
        initial = ::rund::net::ready::create(
            ::rund::net::ready::Config{.max_members = 1u});
        if (!initial.ok()) {
          return;
        }
        initial_destroyed = ::rund::net::ready::destroy(initial.set);
        replacement = ::rund::net::ready::create(
            ::rund::net::ready::Config{.max_members = 1u});
        initial_stale = ::rund::net::ready::clear(initial.set);
        tombstone_stale = ::rund::net::ready::clear(initial_destroyed.set);
        replacement_clear = ::rund::net::ready::clear(replacement.set);
        replacement_destroyed = ::rund::net::ready::destroy(replacement.set);
      });
  READY_SET_ASSERT(reuse_report.ok());
  READY_SET_ASSERT(initial.ok() && initial_destroyed.ok() && replacement.ok());
  READY_SET_ASSERT(initial.set.generation == 1u);
  READY_SET_ASSERT(initial_destroyed.set.id == initial.set.id);
  READY_SET_ASSERT(initial_destroyed.set.generation == 2u);
  READY_SET_ASSERT(replacement.set.id == initial.set.id);
  READY_SET_ASSERT(replacement.set.generation == 3u);
  READY_SET_ASSERT(!initial_stale.ok() && !tombstone_stale.ok());
  READY_SET_ASSERT(initial_stale.code() == ::rund::ReasonCode::TaskInvalid);
  READY_SET_ASSERT(tombstone_stale.code() == ::rund::ReasonCode::TaskInvalid);
  READY_SET_ASSERT(replacement_clear.ok() && replacement_destroyed.ok());

  ::rund::net::ready::Status deterministic_second{};
  ::rund::net::ready::Status deterministic_second_destroyed{};
  const ::rund::Session::Result second_trace =
      deterministic_run(deterministic_second, deterministic_second_destroyed);
  READY_SET_ASSERT(second_trace.ok());
  READY_SET_ASSERT(deterministic_second.ok());
  READY_SET_ASSERT(deterministic_second_destroyed.ok());
  READY_SET_ASSERT(!ReactorReadySetIdentityOwner::same(
      deterministic_first.set, deterministic_second.set));
  READY_SET_ASSERT(first_trace.tasks().trace_hash() ==
                   second_trace.tasks().trace_hash());
  READY_SET_ASSERT(first_trace.tasks().reactor().ready_set_creates() == 1u);
  READY_SET_ASSERT(second_trace.tasks().reactor().ready_set_creates() == 1u);
  READY_SET_ASSERT(first_trace.tasks().reactor().ready_set_destroys() == 1u);
  READY_SET_ASSERT(second_trace.tasks().reactor().ready_set_destroys() == 1u);
  return true;
}

} // namespace rund::node::test_contract
