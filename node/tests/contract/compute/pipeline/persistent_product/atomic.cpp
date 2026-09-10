#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "fixture.hpp"
#include "route.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <thread>

namespace rund_node_test_persistent_product {
namespace {

using namespace std::chrono_literals;

struct PublicationPause final {
  std::mutex gate{};
  std::condition_variable changed{};
  bool entered{};
  bool released{};
};

struct ObserverBarrier final {
  std::mutex gate{};
  std::condition_variable changed{};
  std::size_t started{};

  void arrive() noexcept {
    std::lock_guard lock{gate};
    ++started;
    changed.notify_all();
  }

  [[nodiscard]] bool wait_for(const std::size_t expected) noexcept {
    std::unique_lock lock{gate};
    return changed.wait_for(lock, 20ms, [&] { return started >= expected; });
  }
};

PublicationPause *active_pause{};

void PausePublication() noexcept {
  PublicationPause *const pause = active_pause;
  if (pause == nullptr) {
    return;
  }
  std::unique_lock lock{pause->gate};
  pause->entered = true;
  pause->changed.notify_all();
  pause->changed.wait(lock, [&] { return pause->released; });
}

[[nodiscard]] bool WaitUntilPaused(PublicationPause &pause) noexcept {
  std::unique_lock lock{pause.gate};
  return pause.changed.wait_for(lock, 10s, [&] { return pause.entered; });
}

void Release(PublicationPause &pause) noexcept {
  std::lock_guard lock{pause.gate};
  pause.released = true;
  pause.changed.notify_all();
}

template <typename T>
[[nodiscard]] bool Blocked(std::future<T> &future) noexcept {
  return future.wait_for(20ms) == std::future_status::timeout;
}

[[nodiscard]] bool ProbeAuthority(
    const std::shared_ptr<rund::compute::detail::residency::Pool> &pool,
    const std::shared_ptr<void> &raw_owner) noexcept {
  using rund::compute::detail::sliding_product_detail::SlidingProductOwner;
  const auto owner = std::static_pointer_cast<SlidingProductOwner>(raw_owner);
  if (pool == nullptr || owner == nullptr) {
    return false;
  }
  const auto lease = pool->authority().sliding().begin_execution_sliding(owner->plan);
  // The fused continuation deliberately keeps the live token and owner. A
  // reentrant Authority probe must therefore return Busy immediately, not
  // wait for the continuation's private callback to finish.
  return lease.failure ==
         rund::compute::detail::residency::AuthorityFailure::Busy;
}

} // namespace

bool CheckPersistentPublicationObservers(const rund::compute::Backend backend,
                                         bool &unavailable) noexcept {
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, 9u, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot primary_before =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_before =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  PublicationPause pause{};
  ProductRouteObservation observation{};
  rund::compute::Status status =
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid);
  active_pause = &pause;
  rund::compute::detail::sliding_product_detail::
      inject_persistent_publication_pause_once(PausePublication);
  std::thread writer{[&] {
    status = RunThroughPersistentProductRoute(prepared.state, observation);
  }};
  if (!WaitUntilPaused(pause)) {
    Release(pause);
    writer.join();
    active_pause = nullptr;
    std::fprintf(stderr, "persistent atomic paused=0 started=0 blocked=0 "
                         "authority_ready=0 authority_busy=0 status=0 final=0 "
                         "publication=0 version=0 recovery=0\n");
    return false;
  }

  ObserverBarrier observers{};
  auto primary = std::async(std::launch::async, [&] {
    observers.arrive();
    return SnapshotPublication(prepared.state->pipeline);
  });
  auto alternate = std::async(std::launch::async, [&] {
    observers.arrive();
    return SnapshotPublication(prepared.state->alternate_pipeline);
  });
  auto backing = std::async(std::launch::async, [&] {
    observers.arrive();
    return BackingVersion(*prepared.output);
  });
  auto authority = std::async(std::launch::async, [&] {
    observers.arrive();
    return ProbeAuthority(prepared.state->pipeline->residency_pool,
                          observation.owner);
  });
  const bool observers_started = observers.wait_for(4u);
  const bool blocked = observers_started && Blocked(primary) &&
                       Blocked(alternate) && Blocked(backing);
  const bool authority_ready =
      authority.wait_for(20ms) != std::future_status::timeout;
  Release(pause);
  writer.join();
  active_pause = nullptr;

  const PublicationSnapshot primary_after = primary.get();
  const PublicationSnapshot alternate_after = alternate.get();
  const std::uint64_t version_after = backing.get();
  const bool authority_busy = authority.get();
  const bool final_ok =
      observation.final_received && observation.final.check.ok;
  const bool publication_ok =
      primary_after.generation == primary_before.generation + 1u &&
      primary_after.payload_epoch == primary_before.payload_epoch + 1u &&
      alternate_after.generation == alternate_before.generation + 1u &&
      alternate_after.payload_epoch == alternate_before.payload_epoch + 1u;
  const bool version_ok = version_after == version_before + 1u;
  const bool recovery_ok = BackingRecovery(*prepared.output) == 0u;
  const bool valid = blocked && authority_ready && authority_busy && status &&
                     observation.production_route && final_ok &&
                     publication_ok && version_ok && recovery_ok &&
                     ExactOutput(prepared);
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent atomic started=%u blocked=%u authority_ready=%u "
        "authority_busy=%u status=%u final=%u publication=%u "
        "version=%u/%llu/%llu recovery=%u\n",
        static_cast<unsigned>(observers_started),
        static_cast<unsigned>(blocked), static_cast<unsigned>(authority_ready),
        static_cast<unsigned>(authority_busy),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(final_ok), static_cast<unsigned>(publication_ok),
        static_cast<unsigned>(version_ok),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(version_after),
        static_cast<unsigned>(recovery_ok));
  }
  return valid;
}

} // namespace rund_node_test_persistent_product

#endif
