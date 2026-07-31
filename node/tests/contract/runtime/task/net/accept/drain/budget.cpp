#include "local.hpp"
#include <rund/net/accept.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "../../../coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

int RunAcceptDrainBudgetExhaustedCase() {
  AcceptDrainLoopbackFixture fixture{};
  TEST_ASSERT(PrepareAcceptDrainLoopbackListener(fixture) == 0);
  constexpr std::size_t kClients = 4u;
  constexpr std::uint64_t kBudget = 2u;
  std::array<AcceptDrainSocketCleanup, kClients> clients{};
  TEST_ASSERT(StartAcceptDrainClients(fixture.connect_address, clients) == 0);
  std::vector<rund::net::Socket> accepted_cleanup{};
  accepted_cleanup.reserve(kClients);
  std::uint64_t callback_count = 0u;
  rund::net::accept::Drain warmed{};
  rund::net::accept::Drain drained{};
  std::uint64_t warm_allocations = ~std::uint64_t{0u};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(AcceptDrainRunSpec(), [&] {
    auto drain = [&]() -> rund::task::Task<void> {
      const auto accept = [&](rund::net::accept::Result &&accepted) {
        if (!accepted.ok()) {
          return false;
        }
        accepted_cleanup.emplace_back(std::move(accepted.socket));
        ++callback_count;
        return true;
      };
      const rund::net::accept::Budget budget{
          .max_accepts = static_cast<std::uint32_t>(kBudget)};

      const auto first = co_await rund::net::accept::drain(
          fixture.listener.view(), budget, accept);
      if (first) {
        warmed = *first;
      }

      runtime_task_allocation::Start();
      const auto second = co_await rund::net::accept::drain(
          fixture.listener.view(), budget, accept);
      runtime_task_allocation::Stop();
      warm_allocations = runtime_task_allocation::Count();
      if (second) {
        drained = *second;
      }
    };
    const rund::task::Handle handle =
        rund::task::spawn("accept-drain-budget", drain());
    joined = rund::task::join(handle);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(warmed.ok());
  TEST_ASSERT(warmed.budget_exhausted);
  TEST_ASSERT(!warmed.would_block);
  TEST_ASSERT(!warmed.handler_stopped);
  TEST_ASSERT(warmed.accepts == kBudget);
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.budget_exhausted);
  TEST_ASSERT(!drained.would_block);
  TEST_ASSERT(!drained.handler_stopped);
  TEST_ASSERT(drained.accepts == kBudget);
  TEST_ASSERT(callback_count == kClients);
  TEST_ASSERT(warm_allocations == 0u);
  return 0;
}
