#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/cancel.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/cancel.hpp>

#include <chrono>

NetStatsCancelCase RunNetStatsCancelScenario() {
  NetStatsCancelCase stats{};
  NetStatsSocketPairCleanup cancel_cleanup{};
  NetStatsSocketPairCleanup close_cleanup{};
  stats.cancel_pair_ok = MakeNetStatsSocketPair(cancel_cleanup);
  stats.close_pair_ok = MakeNetStatsSocketPair(close_cleanup);
  if (!stats.cancel_pair_ok || !stats.close_pair_ok) {
    return stats;
  }

  rund::net::Socket cancel_reader =
      rund::node::test::net::admit(cancel_cleanup.left);
  rund::net::Socket cancel_writer =
      rund::node::test::net::admit(cancel_cleanup.right);
  rund::net::Socket close_reader =
      rund::node::test::net::admit(close_cleanup.left);
  rund::net::Socket close_writer =
      rund::node::test::net::admit(close_cleanup.right);
  stats.setup_ok = rund::net::nonblocking(cancel_reader.view(), true).ok() &&
                   rund::net::nonblocking(cancel_writer.view(), true).ok() &&
                   rund::net::nonblocking(close_reader.view(), true).ok() &&
                   rund::net::nonblocking(close_writer.view(), true).ok();
  if (!stats.setup_ok) {
    return stats;
  }

  stats.report = rund::run(NetStatsRunSpec(), [&] {
    auto source = rund::task::stop_source::create();
    stats.source_valid = static_cast<bool>(source);
    if (!stats.source_valid) {
      return 0;
    }
    auto token = source.token();
    stats.token_valid = static_cast<bool>(token);
    if (!stats.token_valid) {
      return 0;
    }

    auto cancel_wait = [&]() -> rund::task::Task<void> {
      stats.cancel_result = co_await rund::net::ready::timed::read(
          cancel_reader.view(), std::chrono::seconds{30}, token);
    };
    auto cancel = [&]() -> rund::task::Task<void> {
      stats.cancel_yielded = co_await rund::task::yield();
      stats.cancel_ok = source.request_stop().ok();
    };
    auto close_wait = [&]() -> rund::task::Task<void> {
      stats.close_wait_result = co_await rund::net::ready::timed::read(
          close_reader.view(), std::chrono::seconds{30});
    };
    auto close = [&]() -> rund::task::Task<void> {
      stats.close_yielded = co_await rund::task::yield();
      stats.close_result = close_reader.close();
    };
    const rund::task::Handle cancel_waiter =
        rund::task::spawn("net-stats-cancel-waiter", cancel_wait());
    const rund::task::Handle cancel_requester =
        rund::task::spawn("net-stats-cancel-requester", cancel());
    const rund::task::Handle close_waiter =
        rund::task::spawn("net-stats-close-waiter", close_wait());
    const rund::task::Handle closer =
        rund::task::spawn("net-stats-close-requester", close());
    stats.joined =
        rund::task::join(cancel_waiter, cancel_requester, close_waiter, closer);
    return 0;
  });
  return stats;
}
