#include "backlog.hpp"
#include "../state/storage.hpp"

#include "../../../reactor/diagnostics.hpp"
#include "../../../reactor/readiness/handle.hpp"
#include "stats.hpp"

#include <algorithm>

namespace rund::node {

bool ReactorBacklogHasReady(const ReactorRuntime &reactor) noexcept {
  return !reactor.ready_backlog.empty();
}

bool ReactorBacklogStoreSuffix(ReactorRuntime &reactor,
                               ::rund::detail::task::StatStorage &stats,
                               const std::vector<ReactorReady> &ordered,
                               const std::size_t consumed) noexcept {
  if (consumed >= ordered.size()) {
    reactor.ready_backlog.clear();
    return true;
  }
  try {
    reactor.ready_backlog.clear();
    reactor.ready_backlog.reserve(ordered.size() - consumed);
    reactor.ready_backlog.insert(reactor.ready_backlog.end(),
                                 ordered.begin() + consumed, ordered.end());
  } catch (...) {
    reactor.ready_backlog.clear();
    return false;
  }
  RecordReactorReadyBacklogPush(reactor.ready_backlog.size());
  RecordReactorBacklogPush(stats, reactor.ready_backlog.size());
  RecordReactorBacklogDepth(stats, reactor.ready_backlog.size());
  return true;
}

bool ReactorBacklogTakePrefix(ReactorRuntime &reactor,
                              ::rund::detail::task::StatStorage &stats,
                              const std::size_t budget,
                              std::vector<ReactorReady> &out) noexcept {
  if (budget == 0u || reactor.ready_backlog.empty()) {
    return false;
  }
  const std::size_t count = std::min(budget, reactor.ready_backlog.size());
  try {
    out.clear();
    out.reserve(count);
    out.insert(out.end(), reactor.ready_backlog.begin(),
               reactor.ready_backlog.begin() + count);
    reactor.ready_backlog.erase(reactor.ready_backlog.begin(),
                                reactor.ready_backlog.begin() + count);
  } catch (...) {
    out.clear();
    return false;
  }
  RecordReactorReadyBacklogDrain(count);
  RecordReactorReadyBacklogScanStepsAvoided(count);
  RecordReactorBacklogDrain(stats, count);
  return true;
}

void ReactorBacklogClear(ReactorRuntime &reactor) noexcept {
  if (!reactor.ready_backlog.empty()) {
    RecordReactorReadyBacklogInvalidation(reactor.ready_backlog.size());
  }
  reactor.ready_backlog.clear();
}

void ReactorBacklogRemoveFd(ReactorRuntime &reactor, const int fd) noexcept {
  const ReactorHandle handle = ReactorHandleFromPublic(fd);
  const std::size_t before = reactor.ready_backlog.size();
  reactor.ready_backlog.erase(
      std::remove_if(
          reactor.ready_backlog.begin(), reactor.ready_backlog.end(),
          [handle](const ReactorReady &ready) { return ready.fd == handle; }),
      reactor.ready_backlog.end());
  const std::size_t removed = before - reactor.ready_backlog.size();
  if (removed != 0u) {
    RecordReactorReadyBacklogInvalidation(removed);
  }
}

void ReactorBacklogRemoveWait(ReactorRuntime &reactor,
                              const std::uint64_t wait_id) noexcept {
  const std::size_t before = reactor.ready_backlog.size();
  reactor.ready_backlog.erase(
      std::remove_if(reactor.ready_backlog.begin(), reactor.ready_backlog.end(),
                     [wait_id](const ReactorReady &ready) {
                       return ready.wait_id == wait_id;
                     }),
      reactor.ready_backlog.end());
  const std::size_t removed = before - reactor.ready_backlog.size();
  if (removed != 0u) {
    RecordReactorReadyBacklogInvalidation(removed);
  }
}

} // namespace rund::node
