#include "local.hpp"

namespace node_accel_contract {

int RunAccelCoreContracts() {
  TEST_ASSERT(RuntimePolicyChoosesOnlyFromLocalEvidence());
  TEST_ASSERT(PublicContextApiContract());
  TEST_ASSERT(ContextRejectsForgedPickOwnerInSupportPaths());
  return 0;
}

}  // namespace node_accel_contract
