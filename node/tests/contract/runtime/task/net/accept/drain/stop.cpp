#include "local.hpp"
#include <rund/net/accept.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

class MoveOnlyAcceptCallback final {
public:
  MoveOnlyAcceptCallback(std::vector<rund::net::Socket> *const accepted,
                         std::uint64_t *const calls,
                         const std::uint64_t stop_after,
                         bool *const owner_alive,
                         bool *const observed_live_owner) noexcept
      : accepted_(accepted), calls_(calls), stop_after_(stop_after),
        owner_alive_(owner_alive), observed_live_owner_(observed_live_owner) {
    *owner_alive_ = true;
  }

  MoveOnlyAcceptCallback(const MoveOnlyAcceptCallback &) = delete;
  MoveOnlyAcceptCallback &operator=(const MoveOnlyAcceptCallback &) = delete;

  MoveOnlyAcceptCallback(MoveOnlyAcceptCallback &&other) noexcept
      : accepted_(other.accepted_), calls_(other.calls_),
        stop_after_(other.stop_after_),
        owner_alive_(std::exchange(other.owner_alive_, nullptr)),
        observed_live_owner_(other.observed_live_owner_) {}

  MoveOnlyAcceptCallback &operator=(MoveOnlyAcceptCallback &&) = delete;

  ~MoveOnlyAcceptCallback() {
    if (owner_alive_ != nullptr) {
      *owner_alive_ = false;
    }
  }

  [[nodiscard]] bool operator()(rund::net::accept::Result &&accepted) {
    *observed_live_owner_ = *owner_alive_;
    if (!accepted.ok()) {
      return false;
    }
    accepted_->emplace_back(std::move(accepted.socket));
    ++*calls_;
    return *calls_ < stop_after_;
  }

private:
  std::vector<rund::net::Socket> *accepted_ = nullptr;
  std::uint64_t *calls_ = nullptr;
  std::uint64_t stop_after_ = 0u;
  bool *owner_alive_ = nullptr;
  bool *observed_live_owner_ = nullptr;
};

} // namespace

int RunAcceptDrainCallbackStopCase() {
  AcceptDrainLoopbackFixture fixture{};
  TEST_ASSERT(PrepareAcceptDrainLoopbackListener(fixture) == 0);
  constexpr std::size_t kClients = 4u;
  constexpr std::uint64_t kStopAfter = 2u;
  std::array<AcceptDrainSocketCleanup, kClients> clients{};
  TEST_ASSERT(StartAcceptDrainClients(fixture.connect_address, clients) == 0);
  std::vector<rund::net::Socket> accepted_cleanup{};
  accepted_cleanup.reserve(kClients);
  std::uint64_t callback_count = 0u;
  bool callback_owner_alive = false;
  bool callback_observed_live_owner = false;
  bool callback_destroyed_after_await = false;
  rund::net::accept::Drain drained{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(AcceptDrainRunSpec(), [&] {
    auto drain = [&]() -> rund::task::Task<void> {
      auto pending = rund::net::accept::drain(
          fixture.listener.view(), rund::net::accept::Budget{.max_accepts = 8u},
          MoveOnlyAcceptCallback{&accepted_cleanup, &callback_count, kStopAfter,
                                 &callback_owner_alive,
                                 &callback_observed_live_owner});
      TEST_ASSERT(callback_owner_alive);
      const auto result = co_await std::move(pending);
      callback_destroyed_after_await = !callback_owner_alive;
      if (result) {
        drained = *result;
      }
    };
    const rund::task::Handle handle =
        rund::task::spawn("accept-drain-stop", drain());
    joined = rund::task::join(handle);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(drained.ok());
  TEST_ASSERT(drained.handler_stopped);
  TEST_ASSERT(!drained.would_block);
  TEST_ASSERT(!drained.budget_exhausted);
  TEST_ASSERT(drained.accepts == kStopAfter);
  TEST_ASSERT(callback_count == kStopAfter);
  TEST_ASSERT(callback_observed_live_owner);
  TEST_ASSERT(callback_destroyed_after_await);
  return 0;
}
