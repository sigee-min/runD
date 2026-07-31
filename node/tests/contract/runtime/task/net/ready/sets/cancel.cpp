#include "local.hpp"
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

namespace rund::node::test_contract {

bool NetReadySetClearCancelsActiveWait() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  rund::net::ready::Set clear_set{};
  rund::net::ready::Set destroy_set{};
  rund::net::ready::Status clear_created{};
  rund::net::ready::Status destroy_created{};
  rund::net::ready::Status clear_add{};
  rund::net::ready::Status destroy_add{};
  rund::net::ready::many::Result clear_wait{};
  rund::net::ready::many::Result destroy_wait{};
  rund::net::ready::Status cleared{};
  rund::net::ready::Status destroyed{};
  rund::task::Status clear_yielded{};
  rund::task::Status destroy_yielded{};
  rund::task::Status joined{};
  std::array<rund::net::ready::Event, 1u> clear_events{};
  std::array<rund::net::ready::Event, 1u> destroy_events{};

  const rund::Session::Result report =
      rund::run(ready_sets::RunSpec(8u, 8u, 8u), [&] {
        clear_created = rund::net::ready::create(
            rund::net::ready::Config{.max_members = 1u});
        destroy_created = rund::net::ready::create(
            rund::net::ready::Config{.max_members = 1u});
        if (!clear_created.ok() || !destroy_created.ok()) {
          return;
        }
        clear_set = clear_created.set;
        destroy_set = destroy_created.set;
        clear_add = rund::net::ready::add(
            clear_set, ready_sets::ReadableRequest(readers[0u].view()));
        destroy_add = rund::net::ready::add(
            destroy_set, ready_sets::ReadableRequest(readers[1u].view()));
        if (!clear_add.ok() || !destroy_add.ok()) {
          return;
        }

        auto wait_clear = [&]() -> rund::task::Task<void> {
          clear_wait = co_await rund::net::ready::many::wait(
              clear_set, std::span<rund::net::ready::Event>{clear_events},
              std::chrono::seconds{30});
        };
        const rund::task::Handle clear_waiter =
            rund::task::spawn("net-ready-set-clear-waiter", wait_clear());
        auto clear = [&]() -> rund::task::Task<void> {
          clear_yielded = co_await rund::task::yield();
          cleared = rund::net::ready::clear(clear_set);
        };
        const rund::task::Handle clearer =
            rund::task::spawn("net-ready-set-clearer", clear());
        auto wait_destroy = [&]() -> rund::task::Task<void> {
          destroy_wait = co_await rund::net::ready::many::wait(
              destroy_set, std::span<rund::net::ready::Event>{destroy_events},
              std::chrono::seconds{30});
        };
        const rund::task::Handle destroy_waiter =
            rund::task::spawn("net-ready-set-destroy-waiter", wait_destroy());
        auto destroy = [&]() -> rund::task::Task<void> {
          destroy_yielded = co_await rund::task::yield();
          destroyed = rund::net::ready::destroy(destroy_set);
        };
        const rund::task::Handle destroyer =
            rund::task::spawn("net-ready-set-destroyer", destroy());
        joined =
            rund::task::join(clear_waiter, clearer, destroy_waiter, destroyer);
      });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(clear_created.ok());
  READY_SET_ASSERT(destroy_created.ok());
  READY_SET_ASSERT(clear_add.ok());
  READY_SET_ASSERT(destroy_add.ok());
  READY_SET_ASSERT(clear_yielded.ok());
  READY_SET_ASSERT(destroy_yielded.ok());
  READY_SET_ASSERT(cleared.ok());
  READY_SET_ASSERT(destroyed.ok());
  READY_SET_ASSERT(!clear_wait.ok());
  READY_SET_ASSERT(clear_wait.code() == rund::ReasonCode::TaskCancelled);
  READY_SET_ASSERT(!destroy_wait.ok());
  READY_SET_ASSERT(destroy_wait.code() == rund::ReasonCode::TaskCancelled);
  return true;
}

} // namespace rund::node::test_contract
