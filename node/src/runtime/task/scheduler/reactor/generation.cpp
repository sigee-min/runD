#include "generation.hpp"

#include "../../../reactor/readiness/handle.hpp"
#include "../../../reactor/readiness/mask.hpp"
#include "../state/storage.hpp"
#include "cleanup/request.hpp"
#include "registration.hpp"
#include "registry.hpp"
#include "stats.hpp"

#include "../../../../host/net/registry/socket.hpp"

#include <algorithm>

namespace rund::node {
namespace {

[[nodiscard]] bool CleanupInvalid(
    Scheduler &scheduler, ReactorRuntime &reactor,
    const std::span<const ReactorWait> stale,
    ReasonCode *const failure, bool *const invalidated) noexcept {
  if (failure != nullptr) {
    *failure = ReasonCode::Ok;
  }
  if (invalidated != nullptr) {
    *invalidated = false;
  }
  for (const ReactorWait &wait : stale) {
    if (wait.fd_generation != 0u) {
      ReactorRegistrationForgetGeneration(reactor, wait.fd,
                                          wait.fd_generation);
    }
  }

  bool cleanup_ok = true;
  for (const ReactorWait &wait : stale) {
    if (ReactorRegistryFindWait(reactor, wait.wait_id) == nullptr) {
      continue;
    }
    if (invalidated != nullptr) {
      *invalidated = true;
    }
    scheduler.RecordReactorObservation(
        task::ObservationKind::IoInvalid, ReasonCode::IoFdInvalid,
        wait.task_id, wait.wait_id, ReactorHandleForPublic(wait.fd),
        ReactorInterestBits(wait.interest),
        ReactorEventBits(ReactorEventsForInterest(wait.interest)));
    if (!scheduler.RecordReactorHostEvent(ReasonCode::IoFdInvalid,
                                          wait.task_id,
                                          wait.host_handle_id)) {
      if (failure != nullptr) {
        *failure = ReasonCode::HostReplayEventMismatch;
      }
      cleanup_ok = false;
      continue;
    }
    if (!ReactorCleanupWait(
            scheduler, ReactorCleanupRequest{
                           .wait_id = wait.wait_id,
                           .group_id = 0u,
                           .reason = ReasonCode::IoFdInvalid,
                           .cancel_timeout_timer = true,
                           .remove_ready_backlog = true,
                           .cleanup_siblings = true,
                           .events = ReactorEventsForInterest(wait.interest),
                           .store_event = true})) {
      if (failure != nullptr && *failure == ReasonCode::Ok) {
        *failure = ReasonCode::IoPollFailed;
      }
      cleanup_ok = false;
    }
  }
  return cleanup_ok;
}

} // namespace

bool ReactorGenerationCollectStaleWaits(
    const ReactorRuntime &reactor, const ReactorHandle fd,
    const std::uint64_t current_generation,
    std::vector<ReactorWait> &stale) noexcept {
  try {
    stale.clear();
    const ReactorFdState *const state = ReactorRegistryFindFd(reactor, fd);
    stale.reserve(state == nullptr ? 0u : state->wait_count);
    for (std::uint32_t slot = ReactorRegistryFirstWait(reactor, fd);
         slot != kNoReactorSlot; slot = ReactorRegistryNextWait(reactor, slot)) {
      const ReactorWait *const wait = ReactorRegistrySlotWait(reactor, slot);
      if (wait == nullptr) {
        stale.clear();
        return false;
      }
      if (wait->fd_generation != 0u &&
          wait->fd_generation != current_generation) {
        stale.push_back(*wait);
      }
    }
  } catch (...) {
    stale.clear();
    return false;
  }
  std::sort(stale.begin(), stale.end(),
            [](const ReactorWait &left, const ReactorWait &right) {
              if (left.wait_id != right.wait_id) {
                return left.wait_id < right.wait_id;
              }
              if (left.task_id != right.task_id) {
                return left.task_id < right.task_id;
              }
              if (left.fd != right.fd) {
                return left.fd < right.fd;
              }
              return left.interest < right.interest;
            });
  return true;
}

bool ReactorGenerationCleanupStaleWaits(Scheduler &scheduler,
                                        const ReactorHandle fd,
                                        const std::uint64_t current_generation,
                                        ReasonCode *const failure) noexcept {
  ReactorRuntime &reactor = scheduler.state_->reactor.reactor;
  std::vector<ReactorWait> &stale = reactor.stale_wait_scratch;
  if (!ReactorGenerationCollectStaleWaits(reactor, fd, current_generation,
                                          stale)) {
    if (failure != nullptr) {
      *failure = ReasonCode::ReactorWaitCapacityExceeded;
    }
    return false;
  }

  return CleanupInvalid(scheduler, reactor, stale, failure, nullptr);
}

bool ReactorGenerationCleanupInvalidWaits(
    Scheduler &scheduler, bool *const invalidated) noexcept {
  ReactorRuntime &reactor = scheduler.state_->reactor.reactor;
  std::vector<ReactorWait> &stale = reactor.stale_wait_scratch;
  try {
    stale.clear();
    const std::size_t count = ReactorRegistrySize(reactor);
    stale.reserve(count);
    for (std::size_t index = 0u; index < count; ++index) {
      const ReactorWait &wait = ReactorRegistryWaitAt(reactor, index);
      if (wait.socket && !::rund::net::IsCurrentSocket(wait.socket)) {
        stale.push_back(wait);
      }
    }
  } catch (...) {
    stale.clear();
    return false;
  }
  std::sort(stale.begin(), stale.end(),
            [](const ReactorWait &left, const ReactorWait &right) {
              return left.wait_id < right.wait_id;
            });
  return CleanupInvalid(scheduler, reactor, stale, nullptr, invalidated);
}

} // namespace rund::node
