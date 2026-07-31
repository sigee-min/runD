#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/socket.hpp>

namespace rund::node::test_contract::ready_sets {

bool PrepareSocketPair(SocketPairCleanup &cleanup, rund::net::Socket &reader,
                       rund::net::Socket &writer) {
  READY_SET_ASSERT(MakeSocketPair(cleanup));
  reader = rund::node::test::net::admit(cleanup.left);
  READY_SET_ASSERT(reader);
  writer = rund::node::test::net::admit(cleanup.right);
  READY_SET_ASSERT(writer);
  READY_SET_ASSERT(rund::node::test::net::generation(reader) != 0u);
  READY_SET_ASSERT(rund::node::test::net::generation(writer) != 0u);
  READY_SET_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  READY_SET_ASSERT(rund::net::nonblocking(writer.view(), true).ok());
  return true;
}

} // namespace rund::node::test_contract::ready_sets
