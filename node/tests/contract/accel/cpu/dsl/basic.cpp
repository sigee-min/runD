#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsDeterministicHashDslOps();
[[nodiscard]] bool RunsDeterministicNoiseDslOps();
[[nodiscard]] bool RunsDeterministicNoiseGridDslOps();
[[nodiscard]] bool RunsDeterministicNormDslOps();
[[nodiscard]] bool RunsDeterministicRangeDslOps();
[[nodiscard]] bool RunsDeterministicMaskDslOps();
[[nodiscard]] bool RunsDeterministicTolDslOps();
[[nodiscard]] bool RunsDeterministicPieceDslOps();
[[nodiscard]] bool RunsDeterministicPolyDslOps();
}  // namespace node_accel_contract

int RunAccelCpuSimdDslBasicContract() {
  TEST_ASSERT(node_accel_contract::RunsDeterministicHashDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicNoiseDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicNoiseGridDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicNormDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicRangeDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicMaskDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicTolDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicPieceDslOps());
  TEST_ASSERT(node_accel_contract::RunsDeterministicPolyDslOps());
  return 0;
}
