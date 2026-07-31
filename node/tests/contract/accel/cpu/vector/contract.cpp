#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool RunsFixedLane32MapWithVectorLaw();
[[nodiscard]] bool RunsFixedLane64MapWithVectorLaw();
[[nodiscard]] bool RunsFixedLane32TailChunkWithVectorLaw();
[[nodiscard]] bool RunsFixedLane32StridedBindingsWithVectorLaw();
[[nodiscard]] bool RunsFixedLane32MultiWriteWithOneEvaluation();
[[nodiscard]] bool CpuSimdFreezesExecutorSelectorsAndScratchBoundary();
} // namespace node_accel_contract

int RunAccelCpuSimdVectorContract() {
  TEST_ASSERT(node_accel_contract::RunsFixedLane32MapWithVectorLaw());
  TEST_ASSERT(node_accel_contract::RunsFixedLane64MapWithVectorLaw());
  TEST_ASSERT(node_accel_contract::RunsFixedLane32TailChunkWithVectorLaw());
  TEST_ASSERT(
      node_accel_contract::RunsFixedLane32StridedBindingsWithVectorLaw());
  TEST_ASSERT(
      node_accel_contract::RunsFixedLane32MultiWriteWithOneEvaluation());
  TEST_ASSERT(
      node_accel_contract::CpuSimdFreezesExecutorSelectorsAndScratchBoundary());
  return 0;
}
