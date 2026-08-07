#include "local.hpp"
#include "src/host/net/test/socket.hpp"
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

  AcceptDrainLoopbackFixture close_fixture{};
  TEST_ASSERT(PrepareAcceptDrainLoopbackListener(close_fixture) == 0);
  constexpr std::size_t kCloseClients = 2u;
  std::array<AcceptDrainSocketCleanup, kCloseClients> close_clients{};
  TEST_ASSERT(StartAcceptDrainClients(close_fixture.connect_address,
                                      close_clients) == 0);
  std::vector<rund::net::Socket> close_accepted{};
  close_accepted.reserve(kCloseClients);
  rund::net::accept::Drain close_drained{};
  rund::net::CloseResult callback_close{};
  std::uint32_t callback_readers = ~std::uint32_t{0u};
  std::uint64_t close_callback_count = 0u;
  rund::task::Status close_joined{};
  const rund::Session::Result close_report =
      rund::run(AcceptDrainRunSpec(), [&] {
        auto close_drain = [&]() -> rund::task::Task<void> {
          const auto result = co_await rund::net::accept::drain(
              close_fixture.listener.view(),
              rund::net::accept::Budget{.max_accepts = 2u},
              [&](rund::net::accept::Result &&accepted) {
                ++close_callback_count;
                callback_readers = rund::node::test::net::reader_count(
                    close_fixture.listener.view());
                if (!accepted.ok() || callback_readers != 0u) {
                  return false;
                }
                close_accepted.emplace_back(std::move(accepted.socket));
                callback_close = close_fixture.listener.close();
                return true;
              });
          if (result) {
            close_drained = *result;
          }
        };
        const rund::task::Handle handle =
            rund::task::spawn("accept-drain-close-listener", close_drain());
        close_joined = rund::task::join(handle);
      });

  TEST_ASSERT(close_report.ok());
  TEST_ASSERT(close_joined.ok());
  TEST_ASSERT(callback_close.ok());
  TEST_ASSERT(!close_fixture.listener);
  TEST_ASSERT(callback_readers == 0u);
  TEST_ASSERT(close_callback_count == 1u);
  TEST_ASSERT(close_accepted.size() == 1u);
  TEST_ASSERT(close_accepted[0u]);
  TEST_ASSERT(!close_drained.ok());
  TEST_ASSERT(close_drained.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(close_drained.accepts == 1u);
  TEST_ASSERT(!close_drained.handler_stopped);
  TEST_ASSERT(!close_drained.would_block);
  TEST_ASSERT(!close_drained.budget_exhausted);
  TEST_ASSERT(close_report.tasks().network().accepts() == 1u);
  TEST_ASSERT(close_report.events().size() >= 2u);
  TEST_ASSERT(close_report.events()[close_report.events().size() - 2u].kind ==
              rund::host::EventKind::NetAccept);
  TEST_ASSERT(close_report.events().back().kind ==
              rund::host::EventKind::IoClose);

  AcceptDrainLoopbackFixture blocking_fixture{};
  TEST_ASSERT(PrepareAcceptDrainLoopbackListener(blocking_fixture) == 0);
  constexpr std::size_t kBlockingClients = 2u;
  std::array<AcceptDrainSocketCleanup, kBlockingClients> blocking_clients{};
  TEST_ASSERT(StartAcceptDrainClients(blocking_fixture.connect_address,
                                      blocking_clients) == 0);
  std::vector<rund::net::Socket> blocking_accepted{};
  blocking_accepted.reserve(kBlockingClients);
  rund::net::accept::Drain blocking_drained{};
  rund::net::NonblockingResult callback_blocking{};
  std::uint64_t blocking_callback_count = 0u;
  rund::task::Status blocking_joined{};
  const rund::Session::Result blocking_report =
      rund::run(AcceptDrainRunSpec(), [&] {
        auto blocking_drain = [&]() -> rund::task::Task<void> {
          const auto result = co_await rund::net::accept::drain(
              blocking_fixture.listener.view(),
              rund::net::accept::Budget{.max_accepts = 2u},
              [&](rund::net::accept::Result &&accepted) {
                if (!accepted.ok()) {
                  return false;
                }
                blocking_accepted.emplace_back(std::move(accepted.socket));
                ++blocking_callback_count;
                callback_blocking = rund::net::nonblocking(
                    blocking_fixture.listener.view(), false);
                return true;
              });
          if (result) {
            blocking_drained = *result;
          }
        };
        const rund::task::Handle handle = rund::task::spawn(
            "accept-drain-blocking-listener", blocking_drain());
        blocking_joined = rund::task::join(handle);
      });

  TEST_ASSERT(blocking_report.ok());
  TEST_ASSERT(blocking_joined.ok());
  TEST_ASSERT(callback_blocking.ok());
  TEST_ASSERT(!callback_blocking.enabled);
  TEST_ASSERT(blocking_fixture.listener);
  TEST_ASSERT(blocking_callback_count == 1u);
  TEST_ASSERT(blocking_accepted.size() == 1u);
  TEST_ASSERT(blocking_accepted[0u]);
  TEST_ASSERT(!blocking_drained.ok());
  TEST_ASSERT(blocking_drained.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(blocking_drained.accepts == 1u);
  TEST_ASSERT(!blocking_drained.handler_stopped);
  TEST_ASSERT(!blocking_drained.would_block);
  TEST_ASSERT(!blocking_drained.budget_exhausted);
  TEST_ASSERT(blocking_report.tasks().network().accepts() == 1u);
  return 0;
}
