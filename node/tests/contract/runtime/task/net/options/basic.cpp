#include "local.hpp"

#include <rund/net/address.hpp>
#include <rund/net/options.hpp>
#include <sys/socket.h>

bool NetSocketOptionsSetAndReadSelectedOptions() {
  NetOptionsSocketCloseGuard guard{};
  NET_OPTIONS_ASSERT(
      OpenNetOptionsInetSocket(rund::net::Transport::Stream, guard));

  const rund::net::option::Result set_reuse = rund::net::option::set(
      guard.socket.view(), rund::net::option::Name::ReuseAddress,
      rund::net::option::Value{.flag = true});
  NET_OPTIONS_ASSERT(set_reuse.ok());
  NET_OPTIONS_ASSERT(set_reuse.option == rund::net::option::Name::ReuseAddress);
  NET_OPTIONS_ASSERT(set_reuse.value.flag);

  const rund::net::option::Result read_reuse = rund::net::option::get(
      guard.socket.view(), rund::net::option::Name::ReuseAddress);
  NET_OPTIONS_ASSERT(read_reuse.ok());
  NET_OPTIONS_ASSERT(read_reuse.option ==
                     rund::net::option::Name::ReuseAddress);
  NET_OPTIONS_ASSERT(read_reuse.value.flag);
  return true;
}
