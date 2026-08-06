#include "access.hpp"
#include "local.hpp"
#include "src/host/net/test/socket.hpp"

#include "test/assert.hpp"

#include <atomic>
#include <cstdint>
#include <rund/net/socket.hpp>

namespace {

[[nodiscard]] bool BurnBound() {
  rund::net::SocketRegistry registry{};
  const rund::node::NativeFdIdentity identity{
      .ok = true,
      .device = 1u,
      .inode = 1u,
      .type = 1u,
  };
  rund::net::SocketSlot *const first = registry.bind(7, identity);
  if (first == nullptr || rund::net::registry::activate(*first) != 1u) {
    return false;
  }
  first->hot.generation.store(rund::net::registry::exhausted - 1u,
                              std::memory_order_release);
  if (rund::net::registry::activate(*first) != 0u ||
      rund::net::registry::load(*first) != rund::net::registry::exhausted) {
    return false;
  }
  registry.release(*first);
  const rund::net::SocketRegistryStats burned = registry.stats();
  return burned.slots == 1u && burned.live == 0u && burned.reusable == 0u &&
         burned.burned == 1u && registry.bind(8, identity) == nullptr &&
         registry.stats().slots == burned.slots;
}

} // namespace

int RunNetRegistryLifetimeGenerationCase() {
  using namespace rund::node::test_contract::net_registry_lifetime;

  SocketPair pair{};
  TEST_ASSERT(MakeSocketPair(pair));
  rund::net::Socket first = rund::node::test::net::admit(pair.left);
  TEST_ASSERT(first);
  int native = rund::node::test::net::native(first);

  constexpr std::uint64_t inactive = rund::net::registry::exhausted - 3u;
  constexpr std::uint64_t last = rund::net::registry::exhausted - 2u;
  TEST_ASSERT(Seed(first.view(), inactive));
  TEST_ASSERT(!rund::net::IsCurrentSocket(first.view()));

  rund::net::Socket admitted = rund::node::test::net::admit(native);
  TEST_ASSERT(admitted);
  TEST_ASSERT(rund::node::test::net::generation(admitted) == last);
  TEST_ASSERT(rund::net::IsCurrentSocket(admitted.view()));
  const int admitted_native = rund::node::test::net::native(admitted);

  TEST_ASSERT(Retire(admitted.view()));
  TEST_ASSERT(Read(first.view()) == rund::net::registry::exhausted - 1u);
  TEST_ASSERT(!rund::net::IsCurrentSocket(admitted.view()));

  TEST_ASSERT(BurnBound());
  TEST_ASSERT(Seed(first.view(), 2u));
  // Fault injection deliberately leaves every Socket generation stale. Hand
  // the still-open descriptor back to the raw pair owner only after that
  // transition so its destructor performs the sole native close.
  pair.left = admitted_native;
  return 0;
}
