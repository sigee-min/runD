#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <rund/net/socket.hpp>
#include <utility>

int RunNetLifecycleOwnerCase() {
  using namespace rund::node::test_contract::net_lifecycle;

  SocketPairCleanup first_pair{};
  SocketPairCleanup second_pair{};
  TEST_ASSERT(MakeSocketPair(first_pair));
  TEST_ASSERT(MakeSocketPair(second_pair));

  rund::net::Socket first = rund::node::test::net::admit(first_pair.left);
  rund::net::Socket second = rund::node::test::net::admit(second_pair.left);
  const rund::net::SocketView first_view = first.view();
  const rund::net::SocketView second_view = second.view();

  second = std::move(first);
  TEST_ASSERT(!first);
  TEST_ASSERT(second);
  TEST_ASSERT(second.view() == first_view);
  TEST_ASSERT(!rund::net::IsCurrentSocket(second_view));
  TEST_ASSERT(rund::net::IsCurrentSocket(first_view));

  const rund::net::CloseResult moved = first.close();
  TEST_ASSERT(!moved);
  TEST_ASSERT(moved.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(second.close());
  TEST_ASSERT(!rund::net::IsCurrentSocket(first_view));

  SocketPairCleanup destructor_pair{};
  TEST_ASSERT(MakeSocketPair(destructor_pair));
  rund::net::SocketView destructor_view{};
  {
    rund::net::Socket owner =
        rund::node::test::net::admit(destructor_pair.left);
    destructor_view = owner.view();
    TEST_ASSERT(rund::net::IsCurrentSocket(destructor_view));
  }
  TEST_ASSERT(!rund::net::IsCurrentSocket(destructor_view));
  return 0;
}
