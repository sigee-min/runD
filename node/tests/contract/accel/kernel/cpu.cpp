#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../cpu/device.hpp"
#include "cpu/local.hpp"
#include "cpu/reject/foreign.hpp"
#include "cpu/reject/scan.hpp"
#include "cpu/scan/run.hpp"
#include "cpu/segmented/reduce.hpp"
#include "cpu/segmented/reduce/range.hpp"
#include "test/assert.hpp"

int RunAccelCpuContextKernelContract() {
  namespace cpu = node_accel_contract::cpu_context;
  const rund::AccelDevice pick = node_accel_contract::PickCpu(cpu::CpuPolicy());
  TEST_ASSERT(pick.check.ok);
  TEST_ASSERT(pick.api == rund::AccelApi::Cpu);
  TEST_ASSERT(cpu::CpuContextRunsMap(pick));
  TEST_ASSERT(cpu::CpuContextRunsScanThenMap(pick));
  TEST_ASSERT(cpu::CpuContextRunsSegmentedScanThenMap(pick));
  TEST_ASSERT(cpu::CpuContextRunsSegmentedReduce(pick));
  TEST_ASSERT(cpu::CpuContextRunsSegmentedReduceRangeDefault(pick));
  TEST_ASSERT(cpu::CpuContextRejectsBadScanHash(pick));
  TEST_ASSERT(cpu::CpuContextRejectsForeignBuffer(pick));
  return 0;
}
