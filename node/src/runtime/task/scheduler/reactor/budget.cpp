#include "budget.hpp"

#include "../../../reactor/diagnostics.hpp"

#include <algorithm>

namespace rund::node {

ReactorBudgetSelection
ReactorBudgetSelect(ReactorRuntime &reactor,
                    const std::vector<ReactorReady> &ordered,
                    const std::size_t budget) noexcept {
  if (budget == 0u) {
    return ReactorBudgetSelection::failed();
  }
  if (ordered.size() <= budget) {
    return ReactorBudgetSelection::selected(ordered);
  }
  try {
    reactor.budget_ready_scratch.clear();
    reactor.budget_ready_scratch.reserve(budget);
    for (std::size_t index = 0u; index < budget; ++index) {
      reactor.budget_ready_scratch.push_back(ordered[index]);
    }
  } catch (...) {
    reactor.budget_ready_scratch.clear();
    return ReactorBudgetSelection::failed();
  }
  RecordReactorReadyBudgetDeferral(ordered.size() - budget);
  return ReactorBudgetSelection::selected(reactor.budget_ready_scratch);
}

std::size_t
ReactorBudgetExtendInvalidFdPrefix(const std::vector<ReactorReady> &ordered,
                                   const std::size_t consumed) noexcept {
  const std::size_t prefix = std::min(consumed, ordered.size());
  std::size_t extended = prefix;
  for (std::size_t seed = 0u; seed < extended; ++seed) {
    if (ordered[seed].disposition != ReactorReadyDisposition::Invalid) {
      continue;
    }
    for (std::size_t index = extended; index < ordered.size(); ++index) {
      if (ordered[seed].fd == ordered[index].fd) {
        extended = index + 1u;
      }
    }
  }
  return extended;
}

} // namespace rund::node
