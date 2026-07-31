#include "local.hpp"
#include "src/host/net/test/socket.hpp"
#include <rund/net/io.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <utility>

#include <unistd.h>

namespace {

using SocketPair = rund::node::test_contract::net_lifecycle::SocketPairCleanup;

enum class CleanupOrder : std::uint8_t {
  WakeFirst,
  AdmissionFirst,
};

struct Scenario {
  rund::net::Socket old{};
  rund::net::Socket current{};
  rund::net::Socket peer{};
  rund::net::SocketView old_view{};
  rund::net::ReceiveResult stale{};
  rund::net::CloseResult closed{};
  rund::net::SendResult sent{};
  rund::net::ReceiveResult received{};
  rund::task::Status joined{};
  rund::task::Status fresh_joined{};
  std::array<std::byte, 1u> payload{std::byte{'g'}};
  std::array<std::byte, 1u> output{};
  std::array<std::byte, 1u> stale_output{};
  std::atomic<bool> started{false};
  std::uint64_t old_id = 0u;
  std::uint64_t old_generation = 0u;
  bool fresh_complete = false;
  bool setup = true;
};

[[nodiscard]] bool HasInvalidEvidence(const rund::Session::Result &report) {
  for (const rund::task::Observation &observation : report.observations()) {
    if (observation.kind != rund::task::ObservationKind::IoInvalid ||
        observation.reason_code != rund::ReasonCode::IoFdInvalid) {
      continue;
    }
    for (const rund::host::Event &event : report.events()) {
      if (event.kind == rund::host::EventKind::IoReady &&
          event.status == rund::host::Status::SyscallFailed &&
          event.task_id == observation.task_id) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] bool Replace(Scenario &scenario, SocketPair &replacement,
                           const int reused_fd, const int old_peer) {
  using namespace rund::node::test_contract::net_lifecycle;

  scenario.closed = scenario.old.close();
  if (!scenario.closed) {
    const std::byte wake{std::byte{'x'}};
    const ssize_t wake_bytes = ::write(old_peer, &wake, sizeof(wake));
    static_cast<void>(wake_bytes);
    return false;
  }
  if (!MakeSocketPair(replacement) || !ForceLeftFd(replacement, reused_fd)) {
    return false;
  }

  scenario.current = rund::node::test::net::admit(replacement.left);
  scenario.peer = rund::node::test::net::admit(replacement.right);
  if (!scenario.current || !scenario.peer ||
      scenario.current.id() != scenario.old_id ||
      rund::node::test::net::generation(scenario.current) == 0u ||
      rund::node::test::net::generation(scenario.current) ==
          scenario.old_generation ||
      !rund::net::nonblocking(scenario.current.view(), true) ||
      !rund::net::nonblocking(scenario.peer.view(), true)) {
    return false;
  }

  scenario.sent = rund::net::direct::send(
      scenario.peer.view(), std::span<const std::byte>{scenario.payload});
  return scenario.sent && scenario.sent.bytes == 1;
}

rund::task::Task<void> ReadCurrent(Scenario &scenario) {
  scenario.received = co_await rund::net::receive(
      scenario.current.view(), std::span<std::byte>{scenario.output});
  scenario.fresh_complete = true;
}

[[nodiscard]] int RunReuse(const CleanupOrder order) {
  using namespace rund::node::test_contract::net_lifecycle;

  SocketPairCleanup original{};
  SocketPairCleanup replacement{};
  TEST_ASSERT(MakeSocketPair(original));
  Scenario scenario{};
  int reused_fd = -1;

  const rund::Session::Result report = rund::run(RunSpec(), [&] {
    scenario.old = rund::node::test::net::admit(original.left);
    reused_fd = rund::node::test::net::native(scenario.old);
    scenario.old_view = scenario.old.view();
    scenario.old_id = scenario.old.id();
    scenario.old_generation = rund::node::test::net::generation(scenario.old);
    scenario.setup = scenario.old && scenario.old_generation != 0u &&
                     rund::net::nonblocking(scenario.old.view(), true);
    if (!scenario.setup) {
      return;
    }

    auto wait_old = [&]() -> rund::task::Task<void> {
      scenario.started.store(true, std::memory_order_release);
      scenario.stale = co_await rund::net::receive(
          scenario.old_view, std::span<std::byte>{scenario.stale_output});
    };
    const rund::task::Handle old_waiter =
        rund::task::spawn("net-reuse-old", wait_old());

    auto replace = [&]() -> rund::task::Task<void> {
      while (!scenario.started.load(std::memory_order_acquire)) {
        static_cast<void>(co_await rund::task::yield());
      }
      std::thread external{[&] {
        scenario.setup =
            Replace(scenario, replacement, reused_fd, original.right);
      }};
      external.join();
      if (scenario.setup && order == CleanupOrder::AdmissionFirst) {
        static_cast<void>(co_await ReadCurrent(scenario));
      }
    };
    const rund::task::Handle replacer =
        rund::task::spawn("net-reuse-replace", replace());
    scenario.joined = rund::task::join(old_waiter, replacer);

    if (scenario.setup && scenario.joined.ok() &&
        order == CleanupOrder::WakeFirst) {
      const rund::task::Handle fresh =
          rund::task::spawn("net-reuse-current", ReadCurrent(scenario));
      scenario.fresh_joined = rund::task::join(fresh);
    }
  });

  TEST_ASSERT(scenario.setup);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(scenario.joined.ok());
  TEST_ASSERT(order == CleanupOrder::AdmissionFirst ||
              scenario.fresh_joined.ok());
  TEST_ASSERT(scenario.fresh_complete);
  TEST_ASSERT(!scenario.stale);
  TEST_ASSERT(scenario.stale.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(scenario.received);
  TEST_ASSERT(scenario.received.bytes == 1);
  TEST_ASSERT(scenario.output == scenario.payload);
  TEST_ASSERT(scenario.stale_output[0] == std::byte{});
  TEST_ASSERT(report.tasks().network().recv_calls() == 1u);
  TEST_ASSERT(HasInvalidEvidence(report));
  return 0;
}

} // namespace

int RunNetLifecycleReuseCase() {
  TEST_ASSERT(RunReuse(CleanupOrder::WakeFirst) == 0);
  TEST_ASSERT(RunReuse(CleanupOrder::AdmissionFirst) == 0);
  return 0;
}
