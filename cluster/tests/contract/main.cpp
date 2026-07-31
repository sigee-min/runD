#include "contract/cases.hpp"

int main() {
  if (const int rc = RunClusterPlacementContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunClusterRunContract(); rc != 0) {
    return rc;
  }
  return RunClusterRetryContract();
}
