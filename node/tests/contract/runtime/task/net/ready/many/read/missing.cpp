#include "src/host/net/test/socket.hpp"
#include "test/assert.hpp"

#include "../local.hpp"

#include <rund/net/ready/many.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>

#include <array>
#include <span>

int RunNetReadyManyReadMissingRuntimeCase() {
  SocketPairCleanup cleanup{};
  TEST_ASSERT(MakeSocketPair(cleanup));
  rund::net::Socket reader = rund::node::test::net::admit(cleanup.left);
  TEST_ASSERT(rund::net::nonblocking(reader.view(), true).ok());
  const std::array<rund::net::ready::Request, 1u> requests{
      rund::net::ready::Request{.socket = reader.view(),
                                .interest =
                                    rund::net::ready::Interest::Readable}};
  std::array<rund::net::ready::Event, 1u> events{};
  const rund::net::ready::many::Result result =
      rund::net::ready::many::wait(
          std::span<const rund::net::ready::Request>{requests},
          std::span<rund::net::ready::Event>{events})
          .wait();
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code() == rund::ReasonCode::NodeRuntimeMissing);
  return 0;
}
