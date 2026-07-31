#include "test/assert.hpp"
#include "pick/run.hpp"

namespace node_accel_contract {
[[nodiscard]] bool CpuPickRunsGenericBackendMap();
[[nodiscard]] bool RejectsForgedArtifactWithoutWritingOutput();
}  // namespace node_accel_contract

int RunAccelCpuSimdBackendContract() {
  TEST_ASSERT(node_accel_contract::CpuPickRunsGenericBackendMap());
  TEST_ASSERT(node_accel_contract::RejectsForgedArtifactWithoutWritingOutput());
  return 0;
}
