#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetRejectsDuplicateMember() {
  ready_sets::SocketPairCleanup cleanup{};
  rund::net::Socket reader{};
  rund::net::Socket writer{};
  READY_SET_ASSERT(ready_sets::PrepareSocketPair(cleanup, reader, writer));

  rund::net::ready::Status created{};
  rund::net::ready::Status first_add{};
  rund::net::ready::Status duplicate_add{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-duplicate", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          if (!created.ok()) {
            return;
          }
          first_add = rund::net::ready::add(
              created.set, ready_sets::ReadableRequest(reader.view()));
          duplicate_add = rund::net::ready::add(
              created.set, ready_sets::ReadableRequest(reader.view()));
          destroyed = rund::net::ready::destroy(created.set);
        });
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(first_add.ok());
  READY_SET_ASSERT(!duplicate_add.ok());
  READY_SET_ASSERT(duplicate_add.code() == rund::ReasonCode::TaskInvalid);
  READY_SET_ASSERT(destroyed.ok());
  READY_SET_ASSERT(report.tasks().reactor().ready_set_members() == 1u);
  return true;
}

} // namespace rund::node::test_contract
