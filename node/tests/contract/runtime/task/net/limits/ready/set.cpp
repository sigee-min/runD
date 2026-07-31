#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetSetCapacityLimitFailsClosed() {
  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_ready_set_capacity = 1u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::ready::Status first{};
  rund::net::ready::Status second{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-ready-set-capacity", [&] {
          first = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          second = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 1u});
          if (first.ok()) {
            destroyed = rund::net::ready::destroy(first.set);
          }
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(first.ok());
  NET_LIMIT_ASSERT(!second.ok());
  NET_LIMIT_ASSERT(second.code() ==
                   rund::ReasonCode::ReactorWaitCapacityExceeded);
  NET_LIMIT_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
