#include "local.hpp"

bool ReadyManyBoundaryBudgetMatches(const ReadyManyBoundaryCase &boundary) {
  return boundary.zero_budget.ok() && boundary.zero_budget.events == 0u &&
         boundary.zero_budget.budget_exhausted;
}
