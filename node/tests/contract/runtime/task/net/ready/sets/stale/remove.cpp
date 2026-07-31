#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetRejectsStaleMemberRemove() {
  ready_sets::SocketPairCleanup cleanup{};
  rund::net::Socket reader{};
  rund::net::Socket writer{};
  READY_SET_ASSERT(ready_sets::PrepareSocketPair(cleanup, reader, writer));

  rund::net::ready::Status created{};
  rund::net::ready::Status added{};
  rund::net::CloseResult closed{};
  rund::net::ready::Status stale_remove{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-stale-remove", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          if (!created.ok()) {
            return;
          }
          added = rund::net::ready::add(
              created.set, ready_sets::ReadableRequest(reader.view()));
          closed = reader.close();
          stale_remove = rund::net::ready::remove(
              created.set, ready_sets::ReadableRequest(reader.view()));
          destroyed = rund::net::ready::destroy(created.set);
        });
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(created.ok());
  READY_SET_ASSERT(added.ok());
  READY_SET_ASSERT(closed.ok());
  READY_SET_ASSERT(!stale_remove.ok());
  READY_SET_ASSERT(stale_remove.code() == rund::ReasonCode::IoFdInvalid);
  READY_SET_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
