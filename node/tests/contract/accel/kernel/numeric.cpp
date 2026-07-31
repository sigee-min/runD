#include <accel/device.hpp>

#include "cpu/local.hpp"
#include "test/assert.hpp"

#include <node/accel/pick.hpp>

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsTransform(const rund::AccelDevice &pick);
[[nodiscard]] bool TransformRejectsNonPowerOfTwo(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunTransformNatively();
[[nodiscard]] bool BackendRunsMatrix(const rund::AccelDevice &pick);
[[nodiscard]] bool MatrixRejectsShapeMismatch(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunMatrixNatively();
[[nodiscard]] bool BackendRunsFactor(const rund::AccelDevice &pick);
[[nodiscard]] bool
NumericAlgebraRejectsOversizeShape(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunFactorNatively();
[[nodiscard]] bool BackendRunsSolve(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsFactorReuseSolve(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSolveNatively();
[[nodiscard]] bool BackendRunsSpectrum(const rund::AccelDevice &pick);
[[nodiscard]] bool SpectrumRejectsInvalidShape(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSpectrumNatively();
[[nodiscard]] bool MetalNumericSourcesUseParallelTopology();
[[nodiscard]] bool VulkanNumericSourcesUseParallelTopology();

} // namespace node_accel_contract

int RunAccelKernelNumericContract() {
  const rund::AccelDevice cpu = rund::node::accel::PickAccel(
      node_accel_contract::cpu_context::CpuPolicy());
  TEST_ASSERT(cpu.check.ok && cpu.api == rund::AccelApi::Cpu);

  TEST_ASSERT(node_accel_contract::BackendRunsTransform(cpu));
  TEST_ASSERT(node_accel_contract::TransformRejectsNonPowerOfTwo(cpu));
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunTransformNatively());
  TEST_ASSERT(node_accel_contract::BackendRunsMatrix(cpu));
  TEST_ASSERT(node_accel_contract::MatrixRejectsShapeMismatch(cpu));
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunMatrixNatively());
  TEST_ASSERT(node_accel_contract::BackendRunsFactor(cpu));
  TEST_ASSERT(node_accel_contract::NumericAlgebraRejectsOversizeShape(cpu));
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunFactorNatively());
  TEST_ASSERT(node_accel_contract::BackendRunsSolve(cpu));
  TEST_ASSERT(node_accel_contract::BackendRunsFactorReuseSolve(cpu));
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunSolveNatively());
  TEST_ASSERT(node_accel_contract::BackendRunsSpectrum(cpu));
  TEST_ASSERT(node_accel_contract::SpectrumRejectsInvalidShape(cpu));
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunSpectrumNatively());
  TEST_ASSERT(node_accel_contract::MetalNumericSourcesUseParallelTopology());
  TEST_ASSERT(node_accel_contract::VulkanNumericSourcesUseParallelTopology());
  return 0;
}
