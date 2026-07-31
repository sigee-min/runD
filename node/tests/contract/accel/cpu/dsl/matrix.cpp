#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsDeterministicLinearDslOps();
[[nodiscard]] bool RunsDeterministicMatrixDslOps();
[[nodiscard]] bool RunsDeterministicAffineDslOps();
[[nodiscard]] bool RunsDeterministicMixDslOps();
[[nodiscard]] bool RunsDeterministicInterpDslOps();
}  // namespace node_accel_contract

int RunAccelCpuSimdDslMatrixContract() {
  TEST_ASSERT(node_accel_contract::RunsDeterministicLinearDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicMatrixDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicAffineDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicMixDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicInterpDslOps());
  return 0;
}
