#include "../local.hpp"
#include <rund/net/accept.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

int RunAcceptDrainWouldBlockCase() {
  AcceptDrainLoopbackFixture fixture{};
  TEST_ASSERT(PrepareAcceptDrainLoopbackListener(fixture) == 0);
  constexpr std::size_t kClients = 4u;
  std::array<AcceptDrainSocketCleanup, kClients> clients{};
  TEST_ASSERT(StartAcceptDrainClients(fixture.connect_address, clients) == 0);
  std::vector<rund::net::Socket> accepted_cleanup{};
  accepted_cleanup.reserve(kClients);
  std::uint64_t callback_count = 0u;
  rund::net::accept::Drain drained{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(AcceptDrainRunSpec(), [&] {
    auto drain = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::accept::drain(
          fixture.listener.view(), rund::net::accept::Budget{.max_accepts = 8u},
          [&](rund::net::accept::Result &&accepted) {
            if (!accepted.ok()) {
              return false;
            }
            accepted_cleanup.emplace_back(std::move(accepted.socket));
            ++callback_count;
            return true;
          });
      if (result) {
        drained = *result;
      }
    };
    const rund::task::Handle handle =
        rund::task::spawn("accept-drain", drain());
    joined = rund::task::join(handle);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.would_block);
  TEST_ASSERT(!drained.budget_exhausted);
  TEST_ASSERT(!drained.handler_stopped);
  TEST_ASSERT(drained.accepts == kClients);
  TEST_ASSERT(callback_count == kClients);
  return 0;
}
