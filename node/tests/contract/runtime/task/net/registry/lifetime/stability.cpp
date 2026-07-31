#include "src/host/net/test/socket.hpp"
#include "access.hpp"
#include "local.hpp"

#include "test/assert.hpp"

#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

namespace {

[[nodiscard]] bool CloseKeepsReservation() {
  using namespace rund::node::test_contract::net_registry_lifetime;

  SocketPair pair{};
  if (!MakeSocketPair(pair)) {
    return false;
  }
  rund::net::Socket socket = rund::node::test::net::admit(pair.left);
  if (!socket) {
    return false;
  }
  int native = rund::node::test::net::native(socket);
  const rund::net::SocketView view = socket.view();
  const rund::net::SocketRegistryStats admitted = Stats();
  if (!rund::net::BeginSocketClose(view) || !HasReservation(view)) {
    return false;
  }
  const rund::net::SocketRegistryStats closing = Stats();
  if (closing.slots != admitted.slots || closing.live != admitted.live) {
    return false;
  }
  SocketPair::Close(native);
  rund::net::FinishSocketClose(view);
  const rund::net::SocketRegistryStats finished = Stats();
  return !HasReservation(view) && finished.slots == admitted.slots &&
         finished.live + 1u == admitted.live;
}

[[nodiscard]] bool CloseHoldsCapacity() {
  using namespace rund::node::test_contract::net_registry_lifetime;

  SocketPair first_pair{};
  SocketPair second_pair{};
  if (!MakeSocketPair(first_pair) || !MakeSocketPair(second_pair)) {
    return false;
  }
  bool scenario_ok = false;
  rund::task::Status joined{};
  rund::SessionConfig config{};
  config.id = 4100u;
  config.workers = 1u;
  config.scheduler.task_capacity = 2u;
  config.scheduler.ready_queue_capacity = 2u;
  config.scheduler.reactor_wait_capacity = 2u;
  config.scheduler.observation_capacity = 8u;
  config.scheduler.host_event_capacity = 8u;
  config.scheduler.net_socket_registry_capacity = 1u;
  const rund::Session::Result report = rund::run(config, [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-registry-close-capacity", [&] {
          rund::net::Socket first =
              rund::node::test::net::admit(first_pair.left);
          if (!first) {
            return;
          }
          int native = rund::node::test::net::native(first);
          const rund::net::SocketView view = first.view();
          const auto reservations = ReservationCounter(view);
          const rund::net::SocketRegistryStats admitted = Stats();
          if (reservations == nullptr ||
              reservations->load(std::memory_order_acquire) != 1u ||
              !rund::net::BeginSocketClose(view)) {
            return;
          }

          const rund::net::SocketAdmission blocked =
              rund::net::AdmitNativeSocket(second_pair.left);
          const rund::net::SocketRegistryStats closing = Stats();
          const bool held =
              !blocked &&
              blocked.code == rund::ReasonCode::TaskCapacityExceeded &&
              reservations->load(std::memory_order_acquire) == 1u &&
              closing.slots == admitted.slots && closing.live == admitted.live;

          SocketPair::Close(native);
          rund::net::FinishSocketClose(view);
          if (!held || reservations->load(std::memory_order_acquire) != 0u) {
            return;
          }

          rund::net::Socket second =
              rund::node::test::net::admit(second_pair.left);
          scenario_ok = second && Stats().slots == admitted.slots &&
                        static_cast<bool>(second.close());
        });
    joined = rund::task::join(task);
  });
  return report && joined && scenario_ok;
}

[[nodiscard]] bool SessionOwnerSurvives() {
  using namespace rund::node::test_contract::net_registry_lifetime;

  SocketPair pair{};
  if (!MakeSocketPair(pair)) {
    return false;
  }
  rund::net::Socket socket{};
  rund::task::Status joined{};
  rund::SessionConfig config{};
  config.id = 4101u;
  config.workers = 1u;
  config.scheduler.task_capacity = 2u;
  config.scheduler.ready_queue_capacity = 2u;
  config.scheduler.reactor_wait_capacity = 2u;
  config.scheduler.observation_capacity = 8u;
  config.scheduler.host_event_capacity = 8u;
  config.scheduler.net_socket_registry_capacity = 1u;
  const rund::Session::Result report = rund::run(config, [&] {
    const rund::task::Handle task =
        rund::task::spawn("net-registry-retained-owner", [&] {
          socket = rund::node::test::net::admit(pair.left);
        });
    joined = rund::task::join(task);
  });
  const std::shared_ptr<std::atomic<std::uint32_t>> reservations =
      ReservationCounter(socket.view());
  if (!report || !joined || !socket || !HasReservation(socket.view()) ||
      reservations == nullptr ||
      reservations->load(std::memory_order_acquire) != 1u) {
    return false;
  }
  const rund::net::SocketView stale = socket.view();
  const rund::net::CloseResult closed = socket.close();
  return closed && !HasReservation(stale) &&
         reservations->load(std::memory_order_acquire) == 0u;
}

} // namespace

int RunNetRegistryLifetimeStabilityCase() {
  using namespace rund::node::test_contract::net_registry_lifetime;

  TEST_ASSERT(CloseKeepsReservation());
  TEST_ASSERT(CloseHoldsCapacity());
  TEST_ASSERT(SessionOwnerSurvives());

  SocketPair pair{};
  TEST_ASSERT(MakeSocketPair(pair));
  rund::net::Socket first = rund::node::test::net::admit(pair.left);
  TEST_ASSERT(first);
  int native = rund::node::test::net::native(first);

  const rund::net::SocketView missing{};
  TEST_ASSERT(!rund::net::IsCurrentSocket(missing));
  TEST_ASSERT(rund::net::IsCurrentSocket(first.view()));

  std::atomic<bool> started{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::atomic<std::uint64_t> checks{0u};
  std::thread reader{[&] {
    if (!rund::net::IsCurrentSocket(first.view())) {
      failed.store(true, std::memory_order_relaxed);
      started.store(true, std::memory_order_release);
      return;
    }
    started.store(true, std::memory_order_release);
    while (!stop.load(std::memory_order_acquire)) {
      if (!rund::net::IsCurrentSocket(first.view())) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
      checks.fetch_add(1u, std::memory_order_relaxed);
    }
  }};
  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  while (!failed.load(std::memory_order_relaxed) &&
         checks.load(std::memory_order_relaxed) == 0u) {
    std::this_thread::yield();
  }
  const rund::net::SocketRegistryStats before_growth = Stats();
  constexpr std::size_t growth_count = 32u;
  std::array<SocketPair, growth_count> growth_pairs{};
  std::array<rund::net::Socket, growth_count> growth_sockets{};
  bool grew = true;
  for (std::size_t i = 0u; i < growth_count; ++i) {
    if (!MakeSocketPair(growth_pairs[i])) {
      grew = false;
      break;
    }
    growth_sockets[i] = rund::node::test::net::admit(growth_pairs[i].left);
    if (!growth_sockets[i]) {
      grew = false;
      break;
    }
  }
  const rund::net::SocketRegistryStats after_growth = Stats();
  stop.store(true, std::memory_order_release);
  reader.join();

  TEST_ASSERT(grew);
  TEST_ASSERT(!failed.load(std::memory_order_relaxed));
  TEST_ASSERT(checks.load(std::memory_order_relaxed) != 0u);
  TEST_ASSERT(rund::net::IsCurrentSocket(first.view()));
  TEST_ASSERT(after_growth.slots >= before_growth.slots);
  TEST_ASSERT(after_growth.slots - before_growth.slots <= growth_count);
  for (auto &socket : growth_sockets) {
    TEST_ASSERT(socket.close());
  }
  TEST_ASSERT(Stats().slots == after_growth.slots);

  TEST_ASSERT(Retire(first.view()));
  TEST_ASSERT(!rund::net::IsCurrentSocket(first.view()));
  rund::net::Socket second = rund::node::test::net::admit(native);
  TEST_ASSERT(second);
  TEST_ASSERT(rund::node::test::net::generation(second) >
              rund::node::test::net::generation(first));
  TEST_ASSERT(!rund::net::IsCurrentSocket(first.view()));
  TEST_ASSERT(rund::net::IsCurrentSocket(second.view()));

  TEST_ASSERT(second.close());
  const rund::net::SocketRegistryStats before_churn = Stats();
  const rund::net::SocketView stale = first.view();
  for (std::size_t iteration = 0u; iteration < 64u; ++iteration) {
    SocketPair churn_pair{};
    TEST_ASSERT(MakeSocketPair(churn_pair));
    rund::net::Socket current = rund::node::test::net::admit(churn_pair.left);
    TEST_ASSERT(current);
    TEST_ASSERT(!rund::net::IsCurrentSocket(stale));
    TEST_ASSERT(current.close());
  }
  const rund::net::SocketRegistryStats after_churn = Stats();
  TEST_ASSERT(after_churn.slots == before_churn.slots);
  TEST_ASSERT(after_churn.burned == before_churn.burned);
  return 0;
}
