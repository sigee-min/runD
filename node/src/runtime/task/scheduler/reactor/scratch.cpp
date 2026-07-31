#include "scratch.hpp"

#include "../../../reactor/diagnostics.hpp"
#include "order.hpp"

#include <algorithm>

namespace rund::node {

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
  reactor.ready.clear();
  reactor.ordered_ready_scratch.clear();
  reactor.budget_ready_scratch.clear();
  reactor.drain_ready_scratch.clear();
  reactor.removed_wait_scratch.clear();
  reactor.stale_wait_scratch.clear();
  reactor.previous_interest_scratch.clear();
}

} // namespace rund::node
