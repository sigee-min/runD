#include "close.hpp"

#include "../../../reactor/readiness/handle.hpp"
#include "../../../reactor/readiness/mask.hpp"
#include "../state/storage.hpp"
#include "cleanup/request.hpp"
#include "registry.hpp"
#include "stats.hpp"

#include <algorithm>

namespace rund::node {
namespace {

[[nodiscard]] bool
ReserveCloseMutationStorage(ReactorRuntime &reactor,
                            std::vector<ReactorWait> &removed) noexcept {
  try {
    removed.clear();
    removed.reserve(ReactorRegistrySize(reactor));
    reactor.changes.reserve(reactor.changes.size() +
                            ReactorRegistryFdCount(reactor));
  } catch (...) {
    removed.clear();
    return false;
  }
  return true;
}

void SortRemovedWaits(std::vector<ReactorWait> &removed) noexcept {
  std::sort(removed.begin(), removed.end(),
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
}

} // namespace

ReasonCode ReactorCloseInvalidateFd(Scheduler &scheduler,
                                    const int fd) noexcept {
  ReactorRuntime &reactor = scheduler.state_->reactor.reactor;
  const ReactorHandle handle = ReactorHandleFromPublic(fd);
  ::rund::detail::task::StatStorage &stats = scheduler.state_->evidence.metrics;
  std::vector<ReactorWait> &removed = reactor.removed_wait_scratch;
  if (!ReserveCloseMutationStorage(reactor, removed)) {
    return ReasonCode::ReactorWaitCapacityExceeded;
  }

  try {
    for (std::uint32_t slot = ReactorRegistryFirstWait(reactor, handle);
         slot != kNoReactorSlot; slot = ReactorRegistryNextWait(reactor, slot)) {
      const ReactorWait *const wait = ReactorRegistrySlotWait(reactor, slot);
      if (wait == nullptr) {
        removed.clear();
        return ReasonCode::IoPollFailed;
      }
      removed.push_back(*wait);
    }
  } catch (...) {
    removed.clear();
    return ReasonCode::ReactorWaitCapacityExceeded;
  }

  SortRemovedWaits(removed);
  ReasonCode result = ReasonCode::Ok;
  for (const ReactorWait &wait : removed) {
    scheduler.RecordReactorObservation(
        task::ObservationKind::IoInvalid, ReasonCode::IoFdInvalid, wait.task_id,
        wait.wait_id, ReactorHandleForPublic(wait.fd),
        ReactorInterestBits(wait.interest),
        ReactorEventBits(ReactorEventsForInterest(wait.interest)));
    (void)scheduler.RecordReactorHostEvent(ReasonCode::IoFdInvalid,
                                           wait.task_id, wait.host_handle_id);
    if (!ReactorCleanupWait(
            scheduler,
            ReactorCleanupRequest{
                .wait_id = wait.wait_id,
                .group_id = 0u,
                .reason = ReasonCode::IoFdInvalid,
                .timeout_cleanup = ReactorTimeoutCleanupPolicy::IfPresent,
                .remove_ready_backlog = true,
                .cleanup_siblings = true,
                .events = ReactorEventsForInterest(wait.interest),
                .store_event = true})) {
      result = ReasonCode::IoPollFailed;
    }
  }
  RecordReactorCloseInvalidatedWaits(stats, removed.size());
  return result;
}

} // namespace rund::node
