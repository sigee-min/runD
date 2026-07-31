#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include "../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <atomic>
#include <cstdint>
#include <rund/net/socket.hpp>
#include <thread>

int RunNetLifecycleLeaseCase() {
  using namespace rund::node::test_contract::net_lifecycle;

  SocketPairCleanup pair{};
  TEST_ASSERT(MakeSocketPair(pair));
  rund::net::Socket owner = rund::node::test::net::admit(pair.left);
  const rund::net::SocketView view = owner.view();

  runtime_task_allocation::Start();
  for (std::uint32_t index = 0u; index < 1024u; ++index) {
    rund::net::SocketLease warm = rund::net::LeaseSocket(view);
    TEST_ASSERT(warm);
  }
  runtime_task_allocation::Stop();
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);

  rund::net::SocketLease lease = rund::net::LeaseSocket(view);
  TEST_ASSERT(lease);
  std::atomic<bool> started{false};
  std::atomic<bool> finished{false};
  rund::net::CloseResult closed{};
  std::thread closer{[&] {
    started.store(true, std::memory_order_release);
    closed = owner.close();
    finished.store(true, std::memory_order_release);
  }};

  while (!started.load(std::memory_order_acquire) ||
         rund::net::IsCurrentSocket(view)) {
    std::this_thread::yield();
  }
  TEST_ASSERT(!finished.load(std::memory_order_acquire));
  TEST_ASSERT(!rund::net::LeaseSocket(view));

  lease = rund::net::SocketLease{};
  closer.join();
  TEST_ASSERT(finished.load(std::memory_order_acquire));
  TEST_ASSERT(closed);
  return 0;
}
