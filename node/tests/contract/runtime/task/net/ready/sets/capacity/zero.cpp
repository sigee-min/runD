#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetRejectsZeroMaxMembers() {
  rund::net::ready::Status created{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(ready_sets::RunSpec(), [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-zero-max-members", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 0u});
        });
    joined = rund::task::join(task);
  });

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(!created.ok());
  READY_SET_ASSERT(created.code() == rund::ReasonCode::TaskInvalid);
  READY_SET_ASSERT(created.set.id == 0u);
  return true;
}

} // namespace rund::node::test_contract
