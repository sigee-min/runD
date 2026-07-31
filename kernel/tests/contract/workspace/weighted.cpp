#include "weighted/local.hpp"

int RunWorkspaceWeightedContract() {
  if (workspace_weighted_contract::Basic() != 0) {
    return 1;
  }
  if (workspace_weighted_contract::Ties() != 0) {
    return 1;
  }
  if (workspace_weighted_contract::Range() != 0) {
    return 1;
  }
  if (workspace_weighted_contract::Source() != 0) {
    return 1;
  }
  if (workspace_weighted_contract::Alias() != 0) {
    return 1;
  }
  return 0;
}
