#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsFixedLane32ScalarOps();
[[nodiscard]] bool CpuSimdUsesFullWideFixedPredicateTruthiness();
[[nodiscard]] bool CpuSimdUsesFullWideFixedLane64PredicateTruthiness();
[[nodiscard]] bool RunsFixedLane32BitOps();
[[nodiscard]] bool RunsFixedLane32ArithmeticOps();
[[nodiscard]] bool RunsFixedLane32NonlinearOps();
[[nodiscard]] bool RunsFixedLane32CompositeDslOps();
[[nodiscard]] bool RunsFixedLane64AllOps();
[[nodiscard]] bool CpuSimdFixedLane32OrderingOpsCovered();
} // namespace node_accel_contract

int RunAccelCpuSimdOpsContract() {
  TEST_ASSERT(node_accel_contract::RunsFixedLane32ScalarOps());
  TEST_ASSERT(
      node_accel_contract::CpuSimdUsesFullWideFixedPredicateTruthiness());
  TEST_ASSERT(
      node_accel_contract::CpuSimdUsesFullWideFixedLane64PredicateTruthiness());
  TEST_ASSERT(node_accel_contract::RunsFixedLane32BitOps());
  TEST_ASSERT(node_accel_contract::RunsFixedLane32ArithmeticOps());
  TEST_ASSERT(node_accel_contract::RunsFixedLane32NonlinearOps());
  TEST_ASSERT(node_accel_contract::RunsFixedLane32CompositeDslOps());
  TEST_ASSERT(node_accel_contract::RunsFixedLane64AllOps());
  TEST_ASSERT(node_accel_contract::CpuSimdFixedLane32OrderingOpsCovered());
  return 0;
}
