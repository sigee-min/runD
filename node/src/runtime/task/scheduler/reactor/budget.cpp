#include "budget.hpp"

#include "../../../reactor/diagnostics.hpp"

namespace rund::node {

ReactorBudgetSelection
ReactorBudgetSelect(ReactorRuntime &reactor,
                    const std::vector<ReactorReady> &ordered,
                    const std::size_t budget) noexcept {
  if (budget == 0u) {
    return ReactorBudgetSelection{
        .ready = nullptr, .consumed = 0u, .ok = false};
  }
  if (ordered.size() <= budget) {
    return ReactorBudgetSelection{
        .ready = &ordered, .consumed = ordered.size(), .ok = true};
  }
  try {
    reactor.budget_ready_scratch.clear();
    reactor.budget_ready_scratch.reserve(budget);
    for (std::size_t index = 0u; index < budget; ++index) {
      reactor.budget_ready_scratch.push_back(ordered[index]);
    }
  } catch (...) {
    reactor.budget_ready_scratch.clear();
    return ReactorBudgetSelection{
        .ready = nullptr, .consumed = 0u, .ok = false};
  }
  RecordReactorReadyBudgetDeferral(ordered.size() - budget);
  return ReactorBudgetSelection{
      .ready = &reactor.budget_ready_scratch, .consumed = budget, .ok = true};
}

} // namespace rund::node
