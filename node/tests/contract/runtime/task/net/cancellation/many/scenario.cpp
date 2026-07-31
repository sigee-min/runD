#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/cancel.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/cancel.hpp>

#include <array>
#include <chrono>
#include <span>

NetCancellationManyCase RunNetCancellationManyScenario() {
  NetCancellationManyCase many{};
  CancelSocketPairCleanup left_cleanup{};
  CancelSocketPairCleanup right_cleanup{};
  many.left_pair_ok = MakeCancelSocketPair(left_cleanup);
  many.right_pair_ok = MakeCancelSocketPair(right_cleanup);
  if (!many.left_pair_ok || !many.right_pair_ok) {
    return many;
  }

  rund::net::Socket left_reader =
      rund::node::test::net::admit(left_cleanup.left);
  rund::net::Socket left_writer =
      rund::node::test::net::admit(left_cleanup.right);
  rund::net::Socket right_reader =
      rund::node::test::net::admit(right_cleanup.left);
  rund::net::Socket right_writer =
      rund::node::test::net::admit(right_cleanup.right);
  many.setup_ok = rund::net::nonblocking(left_reader.view(), true).ok() &&
                  rund::net::nonblocking(left_writer.view(), true).ok() &&
                  rund::net::nonblocking(right_reader.view(), true).ok() &&
                  rund::net::nonblocking(right_writer.view(), true).ok();
  if (!many.setup_ok) {
    return many;
  }

  std::array<rund::net::ready::Request, 2u> requests{
      rund::net::ready::Request{.socket = left_reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
      rund::net::ready::Request{.socket = right_reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable},
  };
  std::array<rund::net::ready::Event, 2u> events{};
  many.report = rund::run(NetCancellationRunSpec(), [&] {
    auto source = rund::task::stop_source::create();
    many.source_valid = static_cast<bool>(source);
    if (!many.source_valid) {
      return;
    }
    auto token = source.token();
    many.token_valid = static_cast<bool>(token);
    if (!many.token_valid) {
      return;
    }

    auto wait = [&]() -> rund::task::Task<void> {
      many.ready = co_await rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events}, std::chrono::seconds{30},
          token);
    };
    auto cancel = [&]() -> rund::task::Task<void> {
      many.cancel_yield = co_await rund::task::yield();
      many.cancel_ok = source.request_stop().ok();
    };
    many.scope = rund::task::scope([&] {
      (void)rund::task::spawn("net-cancel-many-waiter", wait());
      (void)rund::task::spawn("net-cancel-many-requester", cancel());
    });

    many.close = left_reader.close();

    auto sleep = [&]() -> rund::task::Task<void> {
      many.post_close_sleep =
          co_await rund::task::sleep(std::chrono::milliseconds{1});
    };
    many.post_close_scope = rund::task::scope([&] {
      (void)rund::task::spawn("net-cancel-many-short-scope", sleep());
    });
  });
  return many;
}
