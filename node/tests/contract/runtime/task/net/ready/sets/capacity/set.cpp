#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetSetCapacityFailsClosed() {
  rund::net::ready::Status first_set{};
  rund::net::ready::Status second_set{};
  rund::net::ready::Status zero_member_set{};
  rund::task::Status joined{};
  rund::Session runtime{};
  const rund::SessionConfig options = ready_sets::Config(4u, 4u, 8u, 1u, 4u);
  READY_SET_ASSERT(runtime.open(options).ok());
  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-set-capacity", [&] {
          first_set = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          second_set = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          zero_member_set = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 0u});
          if (first_set.ok()) {
            static_cast<void>(rund::net::ready::destroy(first_set.set));
          }
        });
    joined = rund::task::join(task);
  });
  READY_SET_ASSERT(runtime.close().ok());

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(first_set.ok());
  READY_SET_ASSERT(!second_set.ok());
  READY_SET_ASSERT(second_set.code() ==
                   rund::ReasonCode::ReactorWaitCapacityExceeded);
  READY_SET_ASSERT(!zero_member_set.ok());
  READY_SET_ASSERT(zero_member_set.code() == rund::ReasonCode::TaskInvalid);
  return true;
}

} // namespace rund::node::test_contract
