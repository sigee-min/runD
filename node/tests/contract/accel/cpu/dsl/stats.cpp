#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsDeterministicStatsDslOps();
[[nodiscard]] bool RunsDeterministicMomentDslOps();
[[nodiscard]] bool RunsDeterministicCorrDslOps();
}  // namespace node_accel_contract

int RunAccelCpuSimdDslStatsContract() {
  TEST_ASSERT(node_accel_contract::RunsDeterministicStatsDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicMomentDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicCorrDslOps());
  return 0;
}
