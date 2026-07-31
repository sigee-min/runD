#include "stats/local.hpp"

int RunRuntimeTaskNetStatsContract() {
  if (const int rc = NetStatsNestedVisibility(); rc != 0) {
    return rc;
  }
  if (const int rc = NetStatsByteAccounting(); rc != 0) {
    return rc;
  }
  if (const int rc = NetStatsCloseTimeoutCancellation(); rc != 0) {
    return rc;
  }
  return 0;
}
