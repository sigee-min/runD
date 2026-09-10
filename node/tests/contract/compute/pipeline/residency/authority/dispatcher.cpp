#include "local.hpp"

namespace rund_node_test_pipeline_residency {

int CheckAuthorityMigrationGraph() {
  if (const int result = CheckAuthorityMigration(); result != 0) {
    return result;
  }
  if (const int result = CheckAuthorityTransform(); result != 0) {
    return result;
  }
  if (const int result = CheckAuthorityCapacity(); result != 0) {
    return result;
  }
  if (const int result = CheckAuthorityLifetime(); result != 0) {
    return result;
  }
  if (const int result = CheckAuthorityGraph(); result != 0) {
    return result;
  }
  return CheckAuthorityViewCommit();
}

} // namespace rund_node_test_pipeline_residency
