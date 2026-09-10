#include "local.hpp"

#include <rund/net/ready/set.hpp>
#include <rund/task/api.hpp>

#include "src/runtime/task/scheduler/reactor/ready/set/identity.hpp"

namespace rund::node::test_contract {

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
