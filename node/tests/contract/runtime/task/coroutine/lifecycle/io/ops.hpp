#pragma once

#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/await.hpp>

namespace rund::node::test_contract::coroutine {

inline rund::task::Task<void>
ReadyIoAwait(const rund::host::io::FdView read_fd,
             std::atomic<std::uint32_t> *const after_await,
             short *const revents) {
  const rund::task::IoResult ready = co_await rund::host::io::readable(read_fd);
  *revents = ready.revents();
  after_await->fetch_add(1u, std::memory_order_release);
}

inline rund::task::Task<void>
BlockedIoAwait(const rund::host::io::FdView read_fd,
               std::atomic<std::uint32_t> *const after_await,
               short *const revents) {
  const rund::task::IoResult ready = co_await rund::host::io::readable(read_fd);
  *revents = ready.revents();
  after_await->fetch_add(1u, std::memory_order_release);
}

inline rund::task::Task<void> WriteAfterYield(const int write_fd) {
  (void)co_await rund::task::yield();
  const char byte = 'y';
  if (::write(write_fd, &byte, 1u) != 1) {
    co_return;
  }
}

inline rund::task::Task<void>
ReadyManyAwait(const rund::net::SocketView socket,
               std::atomic<std::uint32_t> *const after_await,
               std::uint32_t *const events) {
  const std::array requests{
      rund::net::ready::Request{
          .socket = socket,
          .interest = rund::net::ready::Interest::Readable,
      },
  };
  std::array<rund::net::ready::Event, 1u> ready{};
  const rund::net::ready::many::Result result =
      co_await rund::net::ready::many::wait(requests, ready,
                                            std::chrono::seconds{1});
  *events = result.events;
  after_await->fetch_add(1u, std::memory_order_release);
}

inline rund::task::Task<void>
DiscardReadyOps(const rund::net::SocketView socket,
                std::atomic<std::uint32_t> *const completed) {
  std::array requests{rund::net::ready::Request{
      .socket = socket,
      .interest = rund::net::ready::Interest::Readable,
  }};
  std::array<rund::net::ready::Event, 1u> ready{};
  (void)rund::net::ready::timed::read(socket, std::chrono::seconds{1});
  (void)rund::net::ready::many::wait(requests, ready, std::chrono::seconds{1});
  completed->fetch_add(1u, std::memory_order_release);
  co_return;
}

} // namespace rund::node::test_contract::coroutine
