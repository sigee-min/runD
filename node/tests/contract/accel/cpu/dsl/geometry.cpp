#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsDeterministicVecDslOps();
[[nodiscard]] bool RunsDeterministicSqDslOps();
[[nodiscard]] bool RunsDeterministicMetricDslOps();
[[nodiscard]] bool RunsDeterministicProjDslOps();
[[nodiscard]] bool RunsDeterministicCrossDslOps();
}  // namespace node_accel_contract

int RunAccelCpuSimdDslGeometryContract() {
  TEST_ASSERT(node_accel_contract::RunsDeterministicVecDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicSqDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicMetricDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicProjDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicCrossDslOps());
  return 0;
}
