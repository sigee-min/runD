#pragma once

#include <rund/host.hpp>
#include <rund/net/ready.hpp>
#include <rund/task/await.hpp>

#include <utility>

namespace rund::node::test_contract::reactor {

inline rund::task::Task<void> AwaitReadable(const ::rund::host::io::FdView fd,
                                            task::IoResult *const result) {
  *result = co_await ::rund::host::io::readable(fd);
}

inline rund::task::Task<void> AwaitReadable(const net::SocketView socket,
                                            net::ready::Ticket *const result) {
  *result = co_await net::ready::read(socket);
}

template <class Fn>
rund::task::Task<void> AwaitReadable(const ::rund::host::io::FdView fd,
                                     task::IoResult *const result, Fn after) {
  *result = co_await ::rund::host::io::readable(fd);
  if (result->ok()) {
    std::forward<Fn>(after)();
  }
}

template <class Fn>
rund::task::Task<void> AwaitReadable(const net::SocketView socket,
                                     net::ready::Ticket *const result,
                                     Fn after) {
  *result = co_await net::ready::read(socket);
  if (result->ok()) {
    std::forward<Fn>(after)();
  }
}

} // namespace rund::node::test_contract::reactor
