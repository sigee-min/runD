#include "batch.hpp"

#include "../registry.hpp"

#include <algorithm>

namespace rund::node {
namespace {

[[nodiscard]] bool
CopyRemovedReadyPrefix(std::vector<ReactorReady> &batch_ready,
                       const std::vector<ReactorWait> &removed_waits,
                       const std::vector<ReactorReady> &ordered,
                       const bool all_removed) noexcept {
  try {
    batch_ready.clear();
    batch_ready.reserve(ordered.size());
    batch_ready.insert(batch_ready.end(), ordered.begin(),
                       ordered.begin() + removed_waits.size());
  } catch (...) {
    return false;
  }
  return all_removed && removed_waits.size() == ordered.size();
}

[[nodiscard]] bool
ReserveBatchMutationStorage(ReactorRuntime &reactor,
                            const std::vector<ReactorReady> &ordered,
                            const std::size_t affected_fds) noexcept {
  try {
    reactor.drain_ready_scratch.reserve(ordered.size());
    reactor.removed_wait_scratch.reserve(ordered.size());
    reactor.changes.reserve(reactor.changes.size() + affected_fds);
  } catch (...) {
    return false;
  }
  return true;
}

} // namespace

ReactorDrainBatch
ReactorBuildDrainBatch(ReactorRuntime &reactor,
                       const std::vector<ReactorReady> &ordered) noexcept {
  ReactorDrainBatch batch{};
  std::vector<ReactorFdPreviousInterest> &previous =
      reactor.previous_interest_scratch;
  std::vector<ReactorWait> &removed = reactor.removed_wait_scratch;
  std::vector<ReactorReady> &batch_ready = reactor.drain_ready_scratch;
  if (!ReserveBatchMutationStorage(reactor, ordered, ordered.size())) {
    batch.ok = false;
    return batch;
  }

  const bool removed_all =
      ReactorRegistryRemoveReadyBatch(reactor, ordered, removed, previous);
  if (!removed_all) {
    batch.ok = false;
  }
  if (!CopyRemovedReadyPrefix(batch_ready, removed, ordered, removed_all)) {
    batch.ok = false;
  }

  for (const ReactorFdPreviousInterest &fd : previous) {
    if (!ReactorRegistryCollectChangesForWaitRemove(reactor, fd.fd,
                                                    fd.interest)) {
      batch.ok = false;
    }
  }
  batch.ready = &batch_ready;
  batch.removed_waits = &removed;
  return batch;
}

} // namespace rund::node
