#include "scratch.hpp"

#include "../../../reactor/diagnostics.hpp"
#include "order.hpp"

#include <algorithm>
#include <limits>

namespace rund::node {

bool ReactorScratchPreparePlatformReady(
    ReactorRuntime &reactor, const std::size_t reactor_capacity) noexcept {
  if (reactor_capacity > std::numeric_limits<std::size_t>::max() / 2u) {
    reactor.platform_ready.clear();
    return false;
  }
  try {
    reactor.platform_ready.clear();
    reactor.platform_ready.reserve(reactor_capacity * 2u);
  } catch (...) {
    reactor.platform_ready.clear();
    return false;
  }
  return true;
}

bool ReactorScratchOrderReady(ReactorRuntime &reactor,
                              const std::vector<ReactorReady> &ready) noexcept {
  try {
    reactor.ordered_ready_scratch.clear();
    reactor.ordered_ready_scratch.reserve(ready.size());
    reactor.ordered_ready_scratch.insert(reactor.ordered_ready_scratch.end(),
                                         ready.begin(), ready.end());
    std::sort(reactor.ordered_ready_scratch.begin(),
              reactor.ordered_ready_scratch.end(), ReactorReadyPrecedes);
  } catch (...) {
    reactor.ordered_ready_scratch.clear();
    return false;
  }
  RecordReactorScratchReadyReuse();
  return true;
}

bool ReactorScratchPrepareHostEvents(std::vector<::rund::host::Event> &scratch,
                                     const std::size_t capacity) noexcept {
  try {
    scratch.clear();
    scratch.reserve(capacity);
  } catch (...) {
    scratch.clear();
    return false;
  }
  RecordReactorScratchHostEventReuse();
  return true;
}

void ReactorScratchClear(ReactorRuntime &reactor) noexcept {
  reactor.platform_ready.clear();
  reactor.ready.clear();
  reactor.ordered_ready_scratch.clear();
  reactor.budget_ready_scratch.clear();
  reactor.drain_ready_scratch.clear();
  reactor.removed_wait_scratch.clear();
  reactor.stale_wait_scratch.clear();
  reactor.previous_interest_scratch.clear();
}

} // namespace rund::node
